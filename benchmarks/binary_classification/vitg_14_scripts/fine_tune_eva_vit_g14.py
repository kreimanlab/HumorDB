import numpy as np


# Critical imports
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
import cv2
import copy
import torch
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import shutil
import gc
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import AutoImageProcessor, ViTForImageClassification
from transformers import AutoImageProcessor
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import argparse
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import math
import huggingface_hub
from tqdm.auto import tqdm
import logging
import time
import json
from eva_vit_for_classification import *
from eva_vit import *

from datasets import load_dataset
def transform(example_batch):
    global image_processor
    # Take a list of PIL images and turn them to pixel values
    inputs = image_processor([x for x in example_batch['image']], return_tensors='pt')
    inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs

import numpy as np
from datasets import load_metric

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def get_args():
    parser = argparse.ArgumentParser('Fine Tuning Script for Humor Binary Classification')

    # Model parameters
    parser.add_argument('--experiment_num', default=0, type=int, help='experiment num for logging')
    parser.add_argument('--model_path', default='Salesforce/blip2-opt-2.7b', type=str, metavar='MODEL', help='path to huggingface image processor for ViT/G-14')
    parser.add_argument('--output_dir', type=str, default='vit_g14',
                        help='Output Model Name')
    parser.add_argument('--performance_log', type=str, default='vit_g14/performance_logs', help='directory to log performance')
    parser.add_argument('--LR', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_steps', type=int, default=200, help='Num Steps to Save Model')
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--discard_train_frac",
        type=float,
        default=0.0,
        help="Fraction of discard training data",
    )
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='Dataset/', help='Data directory')

    args = parser.parse_args()
    return args

def load_model(model_path, labels):
    model = ViTgForClassification()
    return model

def load_image_processor(model_path):
    global image_processor
    image_processor  = AutoImageProcessor.from_pretrained(model_path)

if __name__ == '__main__':
    global image_processor
    logger = get_logger(__name__)
    args = get_args()
    print(args.with_tracking)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    random.seed(args.seed)
    accelerator = (
        Accelerator(log_with='tensorboard', project_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    if accelerator.is_main_process:
        if args.output_dir is not None:
            num = random.randint(1, 100)
            # print('Num:', num)
            os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.performance_log, exist_ok=True)
    accelerator.wait_for_everyone()
    load_image_processor(args.model_path)
    train_dataset = load_dataset("imagefolder", data_dir=args.data_dir+"Train/")
    if args.discard_train_frac > 0:
        print(f"Discarding data {args.discard_train_frac * 100} %")
        train_dataset = train_dataset["train"].train_test_split(test_size=args.discard_train_frac)
    print(f"Len of train data {len(train_dataset['train'])}")
    valid_dataset = load_dataset("imagefolder", data_dir=args.data_dir+"Valid/")
    test_dataset = load_dataset("imagefolder", data_dir=args.data_dir+"Test/")
    prepared_train_ds = train_dataset.with_transform(transform)
    prepared_valid_ds = valid_dataset.with_transform(transform)
    prepared_test_ds = test_dataset.with_transform(transform)
    train_dataloader = DataLoader(
        prepared_train_ds['train'], shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size,
    )
    valid_dataloader = DataLoader(prepared_valid_ds['train'], collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(prepared_test_ds['train'], collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
    labels = prepared_train_ds['train'].features['label'].names
    model = load_model(args.model_path, labels)
    convert_weights_to_fp16(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    accelerator.register_for_checkpointing(model)
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.num_epochs * num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_eval = 1000
    best_valid_accuracy = 0
    all_data = {}
    all_data['train_loss'] = []
    all_data['valid_loss'] = []
    all_data['valid_accuracy'] = []
    id = 0
    best_id = 0
    for epoch in range(starting_epoch, args.num_epochs):
        start_time = time.time()
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps % args.save_steps == 0 or completed_steps == args.num_epochs * num_update_steps_per_epoch:
                output_dir = f"step_{completed_steps }"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                if args.with_tracking:
                    all_data['train_loss'].append(total_loss)
                start_valid_time = time.time()
                model.eval()
                samples_seen = 0
                total_loss = 0
                num_items = 0
                all_preds = []
                all_references = []
                for step, batch in enumerate(valid_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    num_items += 1
                    predictions = outputs.logits.argmax(dim=-1)
                    predictions, references = accelerator.gather((predictions, batch["labels"]))
                    # If we are in a multiprocess environment, the last batch has duplicates
                    if accelerator.num_processes > 1:
                        if step == len(valid_dataloader) - 1:
                            predictions = predictions[: len(valid_dataloader.dataset) - samples_seen]
                            references = references[: len(valid_dataloader.dataset) - samples_seen]
                        else:
                            samples_seen += references.shape[0]
                    all_preds.append(predictions.flatten())
                    all_references.append(references.flatten())
                    metric.add_batch(
                        predictions=predictions,
                        references=references,
                    )
                end_valid_time = time.time()
                logger.info(f"Epoch {epoch} evaluation took {end_valid_time-start_valid_time} seconds")
                eval_metric = metric.compute()
                logger.info(f"epoch {epoch}: {eval_metric}")
                total_loss = total_loss / num_items
                with open(os.path.join(args.output_dir, "all_results.json"), "a") as f:
                    json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)
                all_data['valid_loss'].append(total_loss)
                all_data['valid_accuracy'].append(eval_metric['accuracy'])
                if args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    #accelerator.save_state(args.output_dir+'/state')
                    #accelerator.save_model(model, args.output_dir)
                    unwrapped_model = accelerator.unwrap_model(model)
                    # unwrapped_model.save_pretrained(
                    #     args.output_dir+'/model'+str(id), is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    # )
                    # if accelerator.is_main_process:
                    #     image_processor.save_pretrained(args.output_dir+'/model')
                    print(f"Saving model to {args.output_dir+'/model'+str(id)+'.bin'}")
                    torch.save(unwrapped_model.state_dict(), args.output_dir+'/model'+str(id)+'.bin')
                    del unwrapped_model
                    torch.cuda.empty_cache()
                    gc.collect()
                if total_loss < best_eval:
                    if args.output_dir is not None:
                        # accelerator.wait_for_everyone()
                        # accelerator.save_state(args.output_dir+'/state')
                        #accelerator.save_model(model, args.output_dir)
                        #torch.save(model.state_dict(), args.output_dir+'/best_model.bin')
                        best_id = id
                    best_eval = total_loss
                    best_valid_accuracy = eval_metric['accuracy']
                id += 1
            if args.max_train_steps is not None:
                if completed_steps >= args.max_train_steps:
                    break
            del batch
            del outputs
            del loss
            torch.cuda.empty_cache()
            gc.collect()
        end_time = time.time()
        logger.info(f"Epoch {epoch} training took {end_time-start_time} seconds")
        print(f"Epoch {epoch} training took {end_time-start_time} seconds")
        print(f"Best valid model id {best_id}")
    accelerator.wait_for_everyone()
    accelerator.free_memory()
    del model, train_dataloader, valid_dataloader, test_dataloader, optimizer
    # accelerator.load_state(args.output_dir+'/state')
    best_id = 7
    model = ViTgForClassification()
    convert_weights_to_fp16(model)
    state_dict = torch.load(args.output_dir+'/model'+str(best_id)+'.bin')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'module.' in k:
            name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    train_dataloader = DataLoader(
        prepared_train_ds['train'], shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size,
    )
    valid_dataloader = DataLoader(prepared_valid_ds['train'], collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(prepared_test_ds['train'], collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    start_time = time.time()
    model.eval()
    samples_seen = 0
    all_preds = []
    all_labels = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        for i, pred in enumerate(predictions):
            all_preds.append(pred.cpu().detach().item())
            all_labels.append(references[i].cpu().detach().item())
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(test_dataloader) - 1:
                predictions = predictions[: len(test_dataloader.dataset) - samples_seen]
                references = references[: len(test_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    end_time = time.time()
    logger.info(f"test took {end_time-start_time} seconds")
    eval_metric = metric.compute()
    logger.info(f"test accuracy: {eval_metric}")
    print(f"test accuracy: {eval_metric}")
    accelerator.wait_for_everyone()
    with open(os.path.join(args.performance_log, 'performance.txt'), 'a') as f:
        f.write('Experiment '+str(args.experiment_num)+ ': ' + f"test accuracy: {eval_metric['accuracy']} " + f"valid accuracy: {best_valid_accuracy}" + '\n')
    
    progress_bar.close()