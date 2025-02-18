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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import shutil
import gc
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoImageProcessor
from transformers import TrainingArguments
from transformers import Trainer
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

from transformers import AutoModel
class ClassificationOutput():
    def __init__(self, logits=None, loss=None):
        self.logits = logits
        self.loss = loss
def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(torch.bfloat16)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(torch.bfloat16)
class TransformerForRegression(torch.nn.Module):
    def __init__(self, transformer_path, train_full=True):
        super(TransformerForRegression, self).__init__()
        print('Instantiating transformer from:', transformer_path)
        self.transformer = AutoModel.from_pretrained(transformer_path)
        if not train_full:
            for param in self.transformer.parameters():
                param.requires_grad = False
        self.classifier = torch.nn.Linear(self.transformer.config.hidden_size, 1)
    def forward(self, pixel_values= None, labels= None):
        x = pixel_values
        out = self.transformer(x)
        logits = self.classifier(out.pooler_output)
        if labels is not None:
            loss_func = torch.nn.MSELoss()
            loss = loss_func(logits.squeeze(), labels.squeeze())
            if torch.sum(torch.isnan(loss)) > 0:
                print(loss, logits, labels)
                raise('nan')
        return ClassificationOutput(logits=logits, loss=loss)

class FunnyNotFunnyRangeDataset(Dataset):
    def __init__(self, data=[], root_dir=None, transform=None, image_processor=None, labels_path='/range_ratings.csv'):
        label_mapping = {}
        with open(labels_path) as file:
            lines = file.readlines()
            for line in lines:
                words = line.split(',')
                label_mapping[words[0]] = float(words[1])
        if root_dir[-1] != '/':
            root_dir += '/'
        self.root_dir = root_dir
        classes = []
        for file in os.scandir(root_dir):
            if file.is_dir():
                classes.append(file.name)
        data = []
        for i, class_name in enumerate(classes):
            for file in os.listdir(root_dir+class_name):
                label = label_mapping[file]
                data.append((root_dir + class_name + '/'+ file, label))
        self.data = data
        self.num_classes = len(classes)
        self.transform = transform
        self.image_processor = image_processor
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image = Image.open(self.data[index][0])
        if self.image_processor:
            image = self.image_processor(images=image, return_tensors='pt')
            image['pixel_values'] = image['pixel_values'].squeeze(0)
            label = self.data[index][1]
            label_tensor = torch.zeros(1)
            label_tensor[0] = float(label)
            image['labels'] = label_tensor
            image['pixel_values'] = image['pixel_values'].to(torch.bfloat16)
            image['labels'] = image['labels'].to(torch.bfloat16)
            if torch.sum(torch.isnan(image['labels'])) > 0:
                print(self.data[index])
                raise('nan')
            #image['filename'] = self.data[index][0]
            return image
        if not self.image_processor:
            if self.transform:
                try:
                    image = self.transform()(image)
                except:
                    image = self.transform(image)
        label = self.data[index][1]
        label_tensor = torch.zeros(1)
        label_tensor[0] = int(label)
        return {'image_data':image, 'labels':label_tensor, 'filename':self.data[index][0]}

from datasets import load_dataset
def transform(example_batch):
    global image_processor
    # Take a list of PIL images and turn them to pixel values
    inputs = image_processor([x for x in example_batch['image']], return_tensors='pt')

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

def compute_metrics(predictions, references):
    squared_errors = torch.square(predictions - references) 
    mean_squared_error = torch.mean(squared_errors)
    rmse = torch.sqrt(mean_squared_error).item()
    return rmse

def get_args():
    parser = argparse.ArgumentParser('Fine Tuning Script for Humor Binary Classification')

    # Model parameters
    parser.add_argument('--experiment_num', default=0, type=int, help='experiment num for logging')
    parser.add_argument('--model_path', default='facebook/dinov2-large', type=str, metavar='MODEL',
                        help='Path of ViT model to train')
    parser.add_argument('--output_dir', type=str, default='dinov2_large_Funny_Range', 
                        help='Output Model Name')
    parser.add_argument('--performance_log', type=str, default='performance_logs', help='directory to log performance')
    parser.add_argument('--LR', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--cuda_visible_devices', type=str, default='0, 1, 2', help='Visible Devices to Cuda')
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
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='Dataset/', help='Data directory')
    parser.add_argument('--labels_path', type=str, default='range_ratings.csv', help='range labels csv file inside data directory')
    args = parser.parse_args()
    return args

def load_model(model_path):
    model = TransformerForRegression(model_path)
    return model

def load_image_processor(model_path):
    global image_processor
    image_processor  = AutoImageProcessor.from_pretrained(model_path)

if __name__ == '__main__':
    global image_processor
    logger = get_logger(__name__)
    args = get_args()
    print(args.with_tracking)
    random.seed(args.seed)
    accelerator = Accelerator()
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    args.performance_log = args.output_dir + '/' + args.performance_log
    os.makedirs(args.performance_log, exist_ok=True)
    os.makedirs(args.output_dir+'/state', exist_ok=True)
    os.makedirs(args.output_dir+'/model', exist_ok=True)
    accelerator.wait_for_everyone()
    load_image_processor(args.model_path)
    train_dataset = FunnyNotFunnyRangeDataset(root_dir=args.data_dir+'Train', image_processor=image_processor, labels_path=args.labels_path)
    valid_dataset = FunnyNotFunnyRangeDataset(root_dir=args.data_dir+'Valid' ,image_processor=image_processor, labels_path=args.labels_path)
    test_dataset = FunnyNotFunnyRangeDataset(root_dir=args.data_dir+'Test' ,image_processor=image_processor, labels_path=args.labels_path)
    labels = [item[1] for item in train_dataset.data]

    # Round the labels to nearest integer
    rounded_labels = [round(label) for label in labels]

    # Count occurrences of each label
    label_counts = torch.bincount(torch.tensor(rounded_labels).int())

    # Calculate weights
    weights = 1.0 / label_counts.float()
    sample_weights = weights[torch.tensor(rounded_labels).int()]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, sampler=sampler)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True)
    model = load_model(args.model_path)
    model.apply(_convert_weights_to_fp16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.num_epochs * num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_eval = 1000
    best_valid_mse_loss = 0
    all_data = {}
    all_data['train_loss'] = []
    all_data['valid_loss'] = []
    all_data['valid_mse_loss'] = []
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
                    predictions = outputs.logits
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
                end_valid_time = time.time()
                logger.info(f"Epoch {epoch} evaluation took {end_valid_time-start_valid_time} seconds")
                all_preds = torch.cat(all_preds)
                all_references = torch.cat(all_references)
                eval_metric = {'mse_loss':compute_metrics(all_preds.flatten(), all_references.flatten())}
                logger.info(f"epoch {epoch}: {eval_metric}")
                total_loss = total_loss / num_items
                with open(os.path.join(args.output_dir, "all_results.json"), "a") as f:
                    json.dump({"eval_rmse_loss": eval_metric["mse_loss"]}, f)
                accelerator.wait_for_everyone()
                # print(eval_metric, total_loss)
                all_data['valid_loss'].append(total_loss)
                all_data['valid_mse_loss'].append(eval_metric['mse_loss'])
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
                    torch.save(unwrapped_model.state_dict(), args.output_dir+'/model'+str(id)+'.bin')
                    del unwrapped_model
                    torch.cuda.empty_cache()
                    gc.collect()
                if total_loss < best_eval:
                    best_eval = total_loss
                    best_valid_mse_loss = eval_metric['mse_loss']
                    best_id = id
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
    accelerator.wait_for_everyone()
    accelerator.free_memory()
    del model, train_dataloader, valid_dataloader, test_dataloader, optimizer
    #accelerator.load_state(args.output_dir+'/state')
    model = TransformerForRegression(args.model_path)
    model.load_state_dict(torch.load(args.output_dir+'/model'+str(best_id)+'.bin'))
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR, weight_decay=args.weight_decay)
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    start_time = time.time()
    model.eval()
    samples_seen = 0
    all_preds = []
    all_references = []
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(test_dataloader) - 1:
                predictions = predictions[: len(test_dataloader.dataset) - samples_seen]
                references = references[: len(test_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        all_preds.append(predictions.flatten())
        all_references.append(references.flatten())
    end_time = time.time()
    logger.info(f"test took {end_time-start_time} seconds")
    all_preds = torch.cat(all_preds)
    all_references = torch.cat(all_references)
    eval_metric = {'mse_loss':compute_metrics(all_preds.flatten(), all_references.flatten())}
    print(all_preds.flatten(), all_references.flatten())
    logger.info(f"test mse_loss: {eval_metric}")
    print(f"test rmse_loss: {eval_metric}")
    accelerator.wait_for_everyone()
    #print(all_preds.flatten(), all_references.flatten())
    with open(os.path.join(args.performance_log, 'performance.txt'), 'a') as f:
        f.write('Experiment '+str(args.experiment_num)+ ': ' + f"test rmse_loss: {eval_metric['mse_loss']} " + f"valid rmse_loss: {best_valid_mse_loss}" + '\n')