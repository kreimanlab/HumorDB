import numpy as np


# Critical imports
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
import torch
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import gc
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import AutoImageProcessor, BlipForQuestionAnswering, BlipProcessor
from transformers import AutoConfig, AutoModel
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import argparse
from accelerate.logging import get_logger
import math
from tqdm.auto import tqdm
import time
import json
import numpy as np
from datasets import load_metric
from datasets import load_dataset
from accelerate import dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch
from transformers import DefaultDataCollator
from transformers import TrainingArguments
from transformers import Trainer

def transform(example_batch):
    global image_processor
    # Take a list of PIL images and turn them to pixel values
    inputs = image_processor([x for x in example_batch['image']], return_tensors='pt')
    
    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']

    inputs['pixel_values'] = inputs['pixel_values']
    return inputs

class VQADataset(torch.utils.data.Dataset):
    """VQA dataset."""

    def __init__(self, images_folder, processor, image_text, eval_dataset=False):
        self.processor = processor

        self.images = []
        self.labels = []
        self.images_words = []
        num_funny = 0
        num_not_funny = 0
        count_word_funny = 0
        count_word_not_funny = 0
        for file in os.listdir(os.path.join(images_folder, 'Funny')):
            if 'jpg' in file:
                self.images.append(os.path.join(images_folder, 'Funny', file))
                self.labels.append("yes")
                if file in image_text:
                    self.images_words.append(image_text[file])
                    count_word_funny += 1
                else:
                    self.images_words.append(None)
                num_funny += 1
        for file in os.listdir(os.path.join(images_folder, 'Not_Funny')):
            if 'jpg' in file:
                self.images.append(os.path.join(images_folder, 'Not_Funny', file))
                self.labels.append("no")
                if file in image_text:
                    self.images_words.append(image_text[file])
                    count_word_not_funny +=1
                else:
                    self.images_words.append(None)
                num_not_funny += 1
        print(num_funny, num_not_funny, count_word_funny, count_word_not_funny)
        assert len(self.images) == len(self.labels)
        assert len(self.images_words) == len(self.images)
        self.eval = eval_dataset
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # get image + text
        base_question = "Is this image funny?"
        question = ""
        if not self.eval:
            if self.images_words[idx] is not None:
                if random.random() >= 0.0:
                    question = f"The relevant parts of the image are: {','.join(self.images_words[idx])}."
        question = question + base_question
        answer = self.labels[idx]
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        
        encoding = self.processor(images=image, text=question, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor(text=answer, return_tensors="pt").input_ids
        encoding["labels"] = labels
        for k,v in encoding.items():
            encoding[k] = v.squeeze()
        return encoding

def load_model(model_path, labels, pretrained=True):
    global GPU_MAP
    if pretrained:
        model = BlipForQuestionAnswering.from_pretrained(model_path)
        model = model.cuda()
    else:
        config = AutoConfig.from_pretrained(model_path, ignore_mismatched_sizes=True)
        model = BlipForQuestionAnswering._from_config(config)
        model = model.cuda()
        # torch.save(model.state_dict(), 'temp.pth')
        # model = load_checkpoint_and_dispatch(model, 'temp.pth', dtype=torch.float16)
        # os.remove('temp.pth')
    return model, model.device
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

def get_args():
    parser = argparse.ArgumentParser('Fine Tuning Script for Humor Binary Classification')

    # Model parameters
    parser.add_argument('--experiment_num', default=0, type=int, help='experiment num for logging')
    parser.add_argument('--model_path', default='Salesforce/blip-vqa-base', type=str, metavar='MODEL',
                        help='Path of VQA model to train')
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
    parser.add_argument('--word_path', type=str, default='image_common_words.txt', help='Words for images text file')
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='Dataset/', help='Data directory')

    args = parser.parse_args()
    return args

def load_image_processor(model_path):
    global image_processor
    image_processor  = BlipProcessor.from_pretrained(model_path)
    
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
    
    image_text = {}
    with open(args.word_path, 'r') as f:
        for line in f.readlines():
            image, words = line.split(',')
            words_list = words.split(' ')
            image_text[image] = words_list
    
    load_image_processor(args.model_path)
    train_dataset = VQADataset(images_folder=os.path.join(args.data_dir,"Train/"), processor=image_processor, image_text=image_text)
    valid_dataset = VQADataset(images_folder=os.path.join(args.data_dir,"Valid/"), processor=image_processor, image_text=image_text, eval_dataset=True)
    test_dataset = VQADataset(images_folder=os.path.join(args.data_dir,"Test/"), processor=image_processor, image_text=image_text, eval_dataset=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    labels = ['Not_Funny', 'Funny']
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
    all_train_responses = []
    for epoch in range(starting_epoch, args.num_epochs):
        train_loss = 0
        start_time = time.time()
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            
            loss = outputs.loss
            train_loss += loss.detach().float()
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            progress_bar.update(1)
            completed_steps += 1
            # with torch.no_grad():
            #     generated_outputs = model.generate(**batch)
            #     outputs_text = image_processor.batch_decode(generated_outputs)
            #     all_train_responses.append(outputs_text)
            
            if completed_steps % args.save_steps == 0 or completed_steps == args.num_epochs * num_update_steps_per_epoch:
                start_valid_time = time.time()
                model.eval()
                valid_loss = 0
                all_preds = []
                all_references = []
                num_correct = 0
                num_total = 0
                all_responses = []
                for step, batch in enumerate(valid_dataloader):
                    batch = {k: v.to(device=input_device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                        # generated_outputs = model.generate(**batch)
                        # outputs_text = image_processor.batch_decode(generated_outputs, skip_special_tokens=True)
                        # all_responses.append(outputs_text)
                    loss = outputs.loss
                    valid_loss += loss.detach().float()

                # print(all_responses)
                end_valid_time = time.time()
                valid_loss = valid_loss / len(valid_dataloader)
                print(f"Epoch {epoch} evaluation took {end_valid_time-start_valid_time} seconds valid loss: {valid_loss}")
                with open(os.path.join(args.output_dir, "all_results.json"), "a") as f:
                    json.dump({"valid_loss": valid_loss.item()}, f)
                all_data['valid_loss'].append(valid_loss.item())
                if valid_loss < best_eval:
                    if args.output_dir is not None:
                        torch.save(model.state_dict(), os.path.join(args.output_dir, 'model/', str(id)+'.pth'))
                        torch.cuda.empty_cache()
                        gc.collect()
                    best_id = id
                    best_eval = valid_loss
                id += 1
                model.train()
            if args.max_train_steps is not None:
                if completed_steps >= args.max_train_steps:
                    break
            del batch
            del outputs
            del loss
            torch.cuda.empty_cache()
        end_time = time.time()
        train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch} training took {end_time-start_time} seconds, train loss: {train_loss}")
        # print(all_train_responses)
        all_data['train_loss'].append(train_loss)
    gc.collect()
    print(f" best valid loss: {best_eval} best id: {best_id}")
    print("Testing the model", os.path.join(args.output_dir, 'model/', str(best_id)+'.pth'))
    state_dict = torch.load(os.path.join(args.output_dir, 'model/', str(best_id)+'.pth'))
    model.load_state_dict(state_dict)
    start_time = time.time()
    model.eval()
    num_correct = 0
    num_total = 0
    all_preds = []
    all_labels = []
    for step, batch in enumerate(test_dataloader):
        batch = {k: v.to(device=input_device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(**batch)
        outputs_text = image_processor.batch_decode(outputs, skip_special_tokens=True)
        
        preds = torch.tensor([int('yes' in text.lower()) for text in outputs_text]).to(input_device)
        references_text = image_processor.batch_decode(batch['labels'], skip_special_tokens=True)
        references = torch.tensor([int('yes' in text.lower()) for text in references_text]).to(input_device)
        preds, references = preds.flatten(), references.flatten()
        num_correct += (preds == references).sum().item()
        num_total += len(references)
        all_preds.append(outputs_text)
        all_labels.append(references_text)
    
    print(f"num_correct: {num_correct}")
    print(f"num_total: {num_total}")
    print(all_preds)
    print("all_labels", all_labels)
    end_time = time.time()
    print(f"test took {end_time-start_time} seconds")
    test_accuracy = num_correct / num_total
    print(f"test accuracy: {test_accuracy}")
    with open(os.path.join(performance_log_dir, 'performance.txt'), 'a') as f:
        f.write('Experiment '+str(args.experiment_num)+ ': ' + f"test accuracy: {test_accuracy} " + '\n')
    progress_bar.close()
    # training_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     num_train_epochs=args.num_epochs,
    #     per_device_train_batch_size=args.batch_size,
    #     save_steps=args.save_steps,
    #     logging_steps=100,
    #     learning_rate=args.LR,
    #     weight_decay=args.weight_decay,
    #     save_total_limit=2,
    #     remove_unused_columns=False,
    # )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset,
    #     tokenizer=image_processor,
    #     eval_dataset={'valid':valid_dataset, 'test':test_dataset},
    # )
    # trainer.train()