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
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoConfig, AutoModel
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
import numpy as np
from datasets import load_metric
from datasets import load_dataset
from accelerate import dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch

metric = load_metric("accuracy")

GPU_MAP = {0: "4GiB", 1: "8GiB", 2: "8GiB"}

def transform(example_batch):
    global image_processor
    # Take a list of PIL images and turn them to pixel values
    inputs = image_processor([x for x in example_batch['image']], return_tensors='pt')
    
    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']

    inputs['pixel_values'] = inputs['pixel_values']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def get_args():
    parser = argparse.ArgumentParser('Fine Tuning Script for Humor Binary Classification')

    # Model parameters
    parser.add_argument('--experiment_num', default=0, type=int, help='experiment num for logging')
    parser.add_argument('--model_path', default='google/vit-huge-patch14-224-in21k', type=str, metavar='MODEL',
                        help='Path of ViT model to train')
    parser.add_argument('--output_dir', type=str, default='vit_huge_Funny', 
                        help='Output Model Name')
    parser.add_argument('--LR', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_steps', type=int, default=200, help='Num Steps to Save and Eval Model')
    parser.add_argument("--max_train_steps",type=int,default=None,help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument('--pretrained', action='store_true', default=False, help='Use pretrained model or not')
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='Dataset/', help='Data directory')
    parser.add_argument(
        "--discard_train_frac",
        type=float,
        default=0.0,
        help="Fraction of discard training data",
    )

    args = parser.parse_args()
    return args

def load_model(model_path, labels, pretrained=True):
    global GPU_MAP
    if pretrained:
        model = AutoModelForImageClassification.from_pretrained(model_path, num_labels=len(labels), id2label={str(i): c for i, c in  enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}, ignore_mismatched_sizes=True, device_map='auto', max_memory=GPU_MAP)
    else:
        config = AutoConfig.from_pretrained(model_path, num_labels=len(labels), id2label={str(i): c for i, c in  enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)}, ignore_mismatched_sizes=True)
        model = AutoModelForImageClassification.from_config(config)
        torch.save(model.state_dict(), 'temp.pth')
        model = load_checkpoint_and_dispatch(model, 'temp.pth', device_map='auto', max_memory=GPU_MAP)
        os.remove('temp.pth')
    return model, model.device

def load_image_processor(model_path):
    global image_processor
    image_processor  = AutoImageProcessor.from_pretrained(model_path)

if __name__ == '__main__':
    global image_processor
    logger = get_logger(__name__)
    args = get_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    random.seed(args.seed)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    performance_log_dir = os.path.join(args.output_dir, 'performance_log/')
    save_model_dir = os.path.join(args.output_dir, 'model/')
    os.makedirs(performance_log_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)
    
    load_image_processor(args.model_path)
    train_dataset = load_dataset("imagefolder", data_dir=os.path.join(args.data_dir,"Train/"))
    if args.discard_train_frac > 0:
        print(f"Discarding data {args.discard_train_frac * 100} %")
        train_dataset = train_dataset["train"].train_test_split(test_size=args.discard_train_frac)
    print(f"Len of train data {len(train_dataset['train'])}")
    valid_dataset = load_dataset("imagefolder", data_dir=os.path.join(args.data_dir,"Valid/"))
    test_dataset = load_dataset("imagefolder", data_dir=os.path.join(args.data_dir,"Test/"))
    prepared_train_ds = train_dataset.with_transform(transform)
    prepared_valid_ds = valid_dataset.with_transform(transform)
    prepared_test_ds = test_dataset.with_transform(transform)
    train_dataloader = DataLoader(prepared_train_ds['train'], shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=4)
    valid_dataloader = DataLoader(prepared_valid_ds['train'], collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(prepared_test_ds['train'], collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    labels = prepared_train_ds['train'].features['label'].names
    print("Pretrained Model: ", args.pretrained)
    model, input_device = load_model(args.model_path, labels, args.pretrained)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
    total_batch_size = args.batch_size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    progress_bar = tqdm(range(args.num_epochs * num_update_steps_per_epoch))
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
        train_loss = 0
        start_time = time.time()
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(input_device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

            if completed_steps % args.save_steps == 0 or completed_steps == args.num_epochs * num_update_steps_per_epoch:
                start_valid_time = time.time()
                model.eval()
                valid_loss = 0
                all_preds = []
                all_references = []
                num_correct = 0
                num_total = 0
                for step, batch in enumerate(valid_dataloader):
                    batch = {k: v.to(input_device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    loss = outputs.loss
                    valid_loss += loss.detach().float()
                    logits = outputs.logits
                    predictions = outputs.logits.argmax(dim=-1)
                    assert predictions.shape == batch["labels"].shape
                    preds, references = predictions.flatten(), batch["labels"].flatten()
                    all_preds.append(preds)
                    all_references.append(references)
                    num_correct += (preds == references).sum().item()
                    num_total += len(references)
                end_valid_time = time.time()
                valid_accuracy = num_correct / num_total
                valid_loss = valid_loss / len(valid_dataloader)
                print(f"Epoch {epoch} evaluation took {end_valid_time-start_valid_time} seconds valid loss: {valid_loss} valid accuracy: {valid_accuracy}")
                with open(os.path.join(args.output_dir, "all_results.json"), "a") as f:
                    json.dump({"valid_accuracy": valid_accuracy}, f)
                all_data['valid_loss'].append(valid_loss)
                all_data['valid_accuracy'].append(valid_accuracy)
                if valid_loss < best_eval:
                    if args.output_dir is not None:
                        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model/', str(id)+'.pth'))
                        torch.cuda.empty_cache()
                        gc.collect()
                    best_id = id
                    best_eval = valid_loss
                    best_valid_accuracy = valid_accuracy
                id += 1
            if args.max_train_steps is not None:
                if completed_steps >= args.max_train_steps:
                    break
            del batch
            del outputs
            del loss
            torch.cuda.empty_cache()
            # gc.collect()
        end_time = time.time()
        train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch} training took {end_time-start_time} seconds, train loss: {train_loss}")
        all_data['train_loss'].append(train_loss)
    gc.collect()
    print(f" best valid accuracy: {best_valid_accuracy} best id: {best_id}")
    print("Testing the model")
    state_dict = torch.load(os.path.join(args.output_dir, 'model/', str(best_id)+'.pth'))
    model.load_state_dict(state_dict)
    start_time = time.time()
    model.eval()
    num_correct = 0
    num_total = 0
    all_preds = []
    all_labels = []
    for step, batch in enumerate(test_dataloader):
        batch = {k: v.to(input_device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        preds, references = predictions.flatten(), batch["labels"].flatten()
        num_correct += (preds == references).sum().item()
        num_total += len(references)
        all_preds.append(preds)
        all_labels.append(references)
    
    print(f"num_correct: {num_correct}")
    print(f"num_total: {num_total}")
    print(all_preds)
    print("all_labels", all_labels)
    end_time = time.time()
    print(f"test took {end_time-start_time} seconds")
    test_accuracy = num_correct / num_total
    print(f"test accuracy: {test_accuracy}")
    with open(os.path.join(performance_log_dir, 'performance.txt'), 'a') as f:
        f.write('Experiment '+str(args.experiment_num)+ ': ' + f"test accuracy: {test_accuracy} " + f"best valid accuracy: {best_valid_accuracy}" + '\n')
    progress_bar.close()