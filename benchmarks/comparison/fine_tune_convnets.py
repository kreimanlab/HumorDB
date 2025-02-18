# Critical imports
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.nn as nn
import json
import copy

# Dataset class for Siamese Network
class ImagePairDataset(Dataset):
    def __init__(self, root_dir, reps_dir, labels_dict, transform=None):
        if root_dir[-1] != '/':
            root_dir += '/'
        self.root_dir = root_dir
        self.reps_dir = reps_dir
        self.transform = transform
        
        data = []
        reps = []
        
        # Load representative images
        for file in os.listdir(reps_dir):
            reps.append(os.path.join(reps_dir, file))
        
        # Load data pairs
        classes = []
        for file in os.scandir(root_dir):
            if file.is_dir():
                classes.append(file.name)
        print(f"Classes: {classes}")
        for class_name in classes:
            for file in os.listdir(os.path.join(root_dir, class_name)):
                data.append(os.path.join(root_dir, class_name, file))

        print(f"Data: {len(data)}")
        self.pairs_data = []
        self.labels_data = []
        for i in range(len(data)):
            for j in range(len(reps)):
                if labels_dict[f"{data[i].split('/')[-1]}_{reps[j].split('/')[-1]}"] != 0:
                    self.pairs_data.append((data[i], reps[j]))
                    if labels_dict[f"{data[i].split('/')[-1]}_{reps[j].split('/')[-1]}"] < 0:
                        self.labels_data.append(1)
                    else:
                        self.labels_data.append(0)
                # print(labels_dict[f"{data[i].split('/')[-1]}_{reps[j].split('/')[-1]}"], f"{data[i].split('/')[-1]}_{reps[j].split('/')[-1]}")
        print(f"Pairs Data: {len(self.pairs_data)}")
        self.labels_dict = copy.deepcopy(labels_dict)
        for key, val in self.labels_dict.items():
            if val < 0:
                self.labels_dict[key] = 1
            else:
                self.labels_dict[key] = 0
    def __len__(self):
        return len(self.pairs_data)
    
    def __getitem__(self, index):
        img1_path, img2_path = self.pairs_data[index]
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        if self.transform:
            try:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            except:
                img1 = self.transform()(img1)
                img2 = self.transform()(img2)
        
        label = self.labels_dict[f"{img1_path.split('/')[-1]}_{img2_path.split('/')[-1]}"]
        assert label == self.labels_data[index]
        return {'image1': img1, 'image2': img2, 'label': torch.tensor(label, dtype=torch.float32)}

class ResNetAdded(torch.nn.Module):
    def __init__(self, resnet152=None, train_full=True, pretrained=True):
        super(ResNetAdded, self).__init__()
        if resnet152:
            self.resnet152 = resnet152
        else:
            self.resnet152 = models.resnet152(pretrained=pretrained)
        if not train_full:
            for param in self.resnet152.parameters():
                param.requires_grad = False
        num_in_features = self.resnet152.fc.in_features
        self.resnet152.fc = torch.nn.Linear(num_in_features, 1)
        self.feature_extractor = torch.nn.Sequential(*(list(self.resnet152.children())[:-1]))
        self.fc_num_in_features = self.resnet152.fc.in_features

    def forward(self, x):
        """
        x: A batch of images.

        Returns: A tensor of predictions.
        """

        # x = self.resnet50(x)
        # preds = self.classifier(x)
        preds = self.resnet152(x)
        return preds

    def feature_extraction(self, x):
        features = self.feature_extractor(x)
        return features
# Siamese Network model
class SiameseNetwork(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(self.base_model.fc_num_in_features, 1)

    def feature_extraction(self, x):
        x = self.base_model.feature_extraction(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input1, input2):
        output1 = self.feature_extraction(input1)
        output2 = self.feature_extraction(input2)
        diff = torch.abs(output1 - output2)
        out = self.fc(diff)
        return torch.sigmoid(out)

# Load model and transforms function
def load_model_and_transforms(model_name='resnet152', train_full=False, pretrained=True):
    if model_name == 'resnet152':
        base_model = ResNetAdded(train_full=train_full, pretrained=pretrained)
    else:
        raise ValueError('Model name not recognized')

    model = SiameseNetwork(base_model)
    
    if not train_full:
        for param in base_model.parameters():
            param.requires_grad = False
    
    transform = models.ResNet152_Weights.IMAGENET1K_V1.transforms
    
    return model, transform

# Training function
def train(num_epochs, model, train_dataloader, valid_dataloader, save_name, LR=0.001, weight_decay=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    best_valid_loss = float('inf')
    best_valid_epoch = -1
    best_valid_accuracy = 0
    all_train_losses = []
    all_valid_losses = []
    count = 0
    
    for epoch in range(num_epochs):
        print('epoch:', epoch)
        model.train()
        total_loss = 0.0
        correct = 0
        total_size = 0
        
        for data in train_dataloader:
            img1, img2, labels = data['image1'].cuda(), data['image2'].cuda(), data['label'].cuda()
            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels.unsqueeze(1)).float().sum().item()
            total_size += labels.size(0)
        
        train_loss = total_loss / len(train_dataloader)
        train_accuracy = correct / total_size
        all_train_losses.append(train_loss)
        
        model.eval()
        total_loss = 0.0
        correct = 0
        total_size = 0
        
        with torch.no_grad():
            for data in valid_dataloader:
                img1, img2, labels = data['image1'].cuda(), data['image2'].cuda(), data['label'].cuda()
                outputs = model(img1, img2)
                loss = criterion(outputs, labels.unsqueeze(1))
                total_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels.unsqueeze(1)).float().sum().item()
                total_size += labels.size(0)
        
        valid_loss = total_loss / len(valid_dataloader)
        valid_accuracy = correct / total_size
        all_valid_losses.append(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_weights = model.state_dict()
            best_valid_epoch = epoch
            best_valid_accuracy = valid_accuracy
            torch.save(best_model_weights, save_name)
            print(f"Model Saved at Epoch {epoch}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")
    
    plt.plot(all_train_losses, label='Train Loss')
    plt.plot(all_valid_losses, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    return best_valid_accuracy

# Testing function
def test(model, test_dataloader):
    criterion = nn.BCELoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total_size = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_dataloader:
            img1, img2, labels = data['image1'].cuda(), data['image2'].cuda(), data['label'].cuda()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels.unsqueeze(1))
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels.unsqueeze(1)).float().sum().item()
            total_size += labels.size(0)
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    accuracy = correct / total_size
    print(f"Test Loss: {total_loss / len(test_dataloader):.4f}, Test Accuracy: {accuracy:.4f}")
    print(pd.DataFrame(confusion_matrix(all_targets, all_preds), columns=['Pred 0', 'Pred 1'], index=['True 0', 'True 1']))
    
    return accuracy

def get_args():
    parser = argparse.ArgumentParser('Fine Tuning Script for Humor Binary Classification')

    # Model parameters
    parser.add_argument('--experiment_num', default=0, type=int, help='experiment num for logging')
    parser.add_argument('--model', default='resnet152', type=str, metavar='MODEL',
                        help='Name of model to train', choices=['convnext_large', 'resnet152', 'inceptionv3', 'convnext_base'])
    parser.add_argument('--output_file', type=str, default='best_model.bin', 
                        help='Output file name and directory')
    parser.add_argument('--output_dir', type=str, default='resnet152/', help='directory to save model')
    parser.add_argument('--performance_log', type=str, default='performance/', help='directory to log performance')
    parser.add_argument('--train_full', action='store_true', default=False, help='Train full model or just classifier')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Use pretrained model or not')
    parser.add_argument('--LR', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--cuda_visible_devices', type=str, default='0', help='Visible Devices to Cuda')
    parser.add_argument('--labels_path', type=str, default='labels_dict.json', help='Path to labels file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='/home/vedaant/send/NewDataset2/', help='Data directory')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(f"Pretrained: {args.pretrained}")
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    args.performance_log = os.path.join(args.output_dir, args.performance_log)
    os.makedirs(args.performance_log, exist_ok=True)
    model, transform = load_model_and_transforms(model_name=args.model, train_full=args.train_full, pretrained=args.pretrained)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    devices = [int(i) for i in args.cuda_visible_devices.split(',')]
    devices = [i for i in range(len(devices))]
    print('Devices Available: ', devices)
    if len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=devices)
    model = model.cuda()
    labels_dict = json.load(open(args.labels_path, 'r'))
    print('Labels Dict Length:', len(labels_dict), labels_dict['F132.jpg_M637.jpg'])
    train_dataset = ImagePairDataset(root_dir=args.data_dir+'Train/', reps_dir=args.data_dir+'Reps/', labels_dict=labels_dict, transform=transform)
    valid_dataset = ImagePairDataset(root_dir=args.data_dir+'Valid/', reps_dir=args.data_dir+'Reps/', labels_dict=labels_dict, transform=transform)
    test_dataset = ImagePairDataset(root_dir=args.data_dir+'Test/', reps_dir=args.data_dir+'Reps/', labels_dict=labels_dict, transform=transform)
    
    #print len of each dataset
    
    print('Train Dataset Length:', len(train_dataset))
    print('Valid Dataset Length:', len(valid_dataset))
    print('Test Dataset Length:', len(test_dataset))
    
    labels = [item for item in train_dataset.labels_data]

    # Round the labels to nearest integer
    rounded_labels = [round(label) for label in labels]

    # Count occurrences of each label
    label_counts = torch.bincount(torch.tensor(rounded_labels).int())

    # Calculate weights
    weights = 1.0 / label_counts.float()
    sample_weights = weights[torch.tensor(rounded_labels).int()]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    num_1s = sum([1 for i in labels if i == 1])
    num_1s_valid = sum([1 for i in valid_dataset.labels_data if i == 1])
    num_1s_test = sum([1 for i in test_dataset.labels_data if i == 1])
    print('Number of 1s in Train:', num_1s)
    print('Number of 1s in Valid:', num_1s_valid)
    print('Number of 1s in Test:', num_1s_test)
    output_file_path = os.path.join(args.output_dir, args.output_file)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    valid_accuracy = train(num_epochs=args.num_epochs, save_name=output_file_path, model=model, train_dataloader=train_loader, valid_dataloader=valid_loader, LR=args.LR, weight_decay=args.weight_decay)
    
    model.load_state_dict(torch.load(output_file_path))
    test_accuracy = test(model=model, test_dataloader=test_loader)
    val_accuracy = test(model=model, test_dataloader=valid_loader)
    
    print('Test Accuracy:', test_accuracy)
    print('Valid Accuracy:', val_accuracy)
    
    with open(os.path.join(args.performance_log, 'performance.txt'), 'a') as f:
        f.write(f'Experiment {args.experiment_num}: test accuracy: {test_accuracy} valid accuracy: {valid_accuracy}\n')
