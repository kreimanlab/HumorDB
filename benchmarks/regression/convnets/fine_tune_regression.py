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
import shutil
import argparse
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

class FunnyNotFunnyDataset(Dataset):
    def __init__(self, data=[], root_dir=None, transform=None, labels_path='/range_ratings.csv'):
        label_mapping = {}
        with open(labels_path) as file:
            lines = file.readlines()
            for line in lines[1:]:
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
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image = Image.open(self.data[index][0])
        if self.transform:
            try:
                image = self.transform()(image)
            except:
                image = self.transform(image)
        label = self.data[index][1]
        label_tensor = torch.zeros(1)
        label_tensor[0] = label
        return {'image_data':image, 'label':label_tensor, 'filename':self.data[index][0]}

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
        self.feature_extractor = None

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
        if self.feature_extractor is None:
            self.feature_extractor = torch.nn.Sequential(*(list(self.resnet152.children())[:-1]))
        features = self.feature_extractor(x)
        return features

class InceptionV3Added(torch.nn.Module):
    def __init__(self, inception=None, train_full=True, pretrained=True):
        super(InceptionV3Added, self).__init__()
        if inception:
            self.xception = inception
        else:
            self.inception = models.inception_v3(pretrained=pretrained)
        if not train_full:
            for param in self.inception.parameters():
                param.requires_grad = False
        self.inception.fc = torch.nn.Linear(2048, 1)
        self.feature_extractor = None
    def forward(self, x):
        """
        x: A batch of images.

        Returns: A tensor of predictions.
        """
        preds = self.inception(x)
        return preds

class ConvNextAdded(torch.nn.Module):
    def __init__(self, convnext=None, train_full=True, pretrained=True, in_features_to_classifier=1024):
        super(ConvNextAdded, self).__init__()
        if convnext:
            self.convnext = convnext
        else:
            self.convnext = models.convnext_base(pretrained=pretrained)
        if pretrained:
            if not train_full:
                for param in self.convnext.parameters():
                    param.requires_grad = False
        self.convnext.classifier[2] = torch.nn.Linear(in_features_to_classifier, 1)
        for param in self.convnext.classifier.parameters():
            param.requires_grad = True
        self.feature_extractor = None
    def forward(self, x):
        """
        x: A batch of images.

        Returns: A tensor of predictions.
        """
        preds = self.convnext(x)
        return preds
def get_args():
    parser = argparse.ArgumentParser('Fine Tuning Script for Humor Binary Classification')

    # Model parameters
    parser.add_argument('--experiment_num', default=0, type=int, help='experiment num for logging')
    parser.add_argument('--model', default='convnext_large', type=str, metavar='MODEL',
                        help='Name of model to train', choices=['convnext_large', 'resnet152', 'inceptionv3', 'convnext_base'])
    parser.add_argument('--output_dir', type=str, default='convnext_large', 
                        help='Output file name and directory')
    parser.add_argument('--output_file', type=str, default='best_model.bin', 
                        help='Output file name and directory')
    parser.add_argument('--performance_log', type=str, default='performance', help='directory to log performance')
    parser.add_argument('--train_full', action='store_true', default=False, help='Train full model or just classifier')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained model or not')
    parser.add_argument('--LR', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--cuda_visible_devices', type=str, default='0,1', help='Visible Devices to Cuda')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='Dataset/', help='Data directory')
    parser.add_argument('--labels_path', type=str, default='range_ratings.csv', help='range labels csv file inside data directory')
    args = parser.parse_args()
    return args

def load_model_and_transforms(model_name='convnext_large', train_full=False, pretrained=True):
    if model_name == 'convnext_large':
        model = ConvNextAdded(convnext=models.convnext_large(pretrained=pretrained), train_full=train_full, in_features_to_classifier=1536)
        transform = models.ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms
    elif model_name == 'resnet152':
        model = ResNetAdded(pretrained=pretrained, train_full=train_full)
        transform = models.ResNet152_Weights.IMAGENET1K_V1.transforms
    elif model_name == 'inceptionv3':
        model = InceptionV3Added(pretrained=pretrained, train_full=train_full)
        transform = models.Inception_V3_Weights.IMAGENET1K_V1.transforms
    elif model_name == 'convnext_base':
        model = ConvNextAdded(pretrained=pretrained, train_full=train_full)
        transform = models.ConvNeXt_Base_Weights.IMAGENET1K_V1.transforms
    else:
        raise ValueError('Model name not recognized')
    return model, transform

def train(num_epochs, model, train_dataloader, valid_dataloader, save_name, LR=0.001, weight_decay=0.001):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    best_valid_loss = 10000
    best_valid_epoch = -1
    best_valid_accuracy = 0
    all_train_losses = []
    all_valid_losses = []
    count = 0
    for epoch in range(num_epochs):
        print('epoch:', epoch)
        model.train()
        running_loss = 0.0
        total_loss = []
        lrs = []
        total_size = 0
        correct = 0
        for i, data in enumerate(train_dataloader):
            inputs = data['image_data'].cuda()
            label = data['label'].cuda()
            inputs = inputs.type(torch.cuda.FloatTensor)
            label = label.type(torch.cuda.FloatTensor)
            output = model.forward(inputs)
            batch_size = inputs.size(0)
            gc.collect()
            del inputs
            loss = criterion(output, label)
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_rmse = torch.sqrt(loss)
            total_loss.append(loss_rmse.item())
            gc.collect()
            del loss
            del loss_rmse
            torch.cuda.empty_cache()
            gc.collect()
            #gpu_usage()
            del predictions
            del label
            del output
            #gpu_usage()
            torch.cuda.empty_cache()
            #print(predictions, "\n", targets, "\n", correct)
            total_size += batch_size
            #print(correct, total_size)
        print('Mean Train loss:', np.mean(total_loss))
        all_train_losses.append(np.mean(total_loss))
        model.eval()
        total_size = 0
        total_loss = []
        correct = 0
        with torch.no_grad():
            for data in valid_dataloader:
                gc.collect()
                torch.cuda.empty_cache()
                inputs = data["image_data"].cuda()
                targets = data["label"].cuda()

                inputs = inputs.type(torch.cuda.FloatTensor)
                targets = targets.type(torch.cuda.FloatTensor)
                #print(ids.shape, "ids")
                batch_size = inputs.size(0)

                output = model.forward(inputs)
                gc.collect()
                del inputs
                loss = criterion(output, targets)
                loss_rmse = torch.sqrt(loss)
                total_loss.append(loss_rmse.item())
                gc.collect()
                del loss
                del loss_rmse
                torch.cuda.empty_cache()
                gc.collect()
                del predictions
                del targets
                del output
                torch.cuda.empty_cache()
                total_size += batch_size
                #gpu_usage()
        all_valid_losses.append(np.mean(total_loss))
        if np.mean(total_loss) < best_valid_loss:
            best_valid_loss = np.mean(total_loss)
            best_model_weights = copy.deepcopy(model.state_dict())
            best_valid_epoch = epoch
            path = save_name
            torch.save(model.state_dict(), path)
            print(f"Model Saved")
        if len(all_valid_losses) > 1:
            if np.abs(all_valid_losses[-1] - all_valid_losses[-2]) > 0.004:
                count = 0
            else:
                count += 1
        print("Validation Loss over a batch: {:.4f}; Best Validation Loss: {:.2f}%".format(np.mean(total_loss), best_valid_loss))
        if count > 3:
            break
    print('Best Validation Loss RMSE:', best_valid_loss, 'Best Validation Epoch:', best_valid_epoch)
    epochs = [i+1 for i in range(len(all_train_losses))]
    plt.plot(epochs, all_train_losses, marker='o')
    plt.plot(epochs, all_valid_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    return best_valid_loss

def test(model, test_dataloader):
    criterion = torch.nn.MSELoss()
    model.eval()
    total_size = 0
    total_loss = []
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in test_dataloader:
            gc.collect()
            torch.cuda.empty_cache()
            inputs = data["image_data"].cuda()
            targets = data["label"].cuda()

            inputs = inputs.type(torch.cuda.FloatTensor)
            targets = targets.type(torch.cuda.FloatTensor)
            #print(ids.shape, "ids")
            batch_size = inputs.size(0)

            output = model.forward(inputs)
            gc.collect()
            del inputs
            loss = criterion(output, targets)
            loss_rmse = torch.sqrt(loss)
            total_loss.append(loss_rmse.item())
            gc.collect()
            del loss
            torch.cuda.empty_cache()
            gc.collect()
            del predictions
            del targets
            del output
            torch.cuda.empty_cache()
            total_size += batch_size
          #gpu_usage()
    print("Total Test Loss: {:.4f}; Mean Test Loss: {:.4f}%".format(np.sum(total_loss), np.mean(total_loss)))
    #print(pd.DataFrame(confusion_matrix(all_preds, all_targets), columns=['Test Not Funny', 'Test Funny'], index=['Pred Not Funny', 'Pred Funny']))
    return np.mean(total_loss)

if __name__ == '__main__':
    args = get_args()
    random.seed(args.seed)
    model, transform = load_model_and_transforms(model_name=args.model, train_full=args.train_full, pretrained=args.pretrained)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    args.performance_log = args.output_dir + '/' + args.performance_log
    args.output_file = args.output_dir + '/' + args.output_file
    os.makedirs(args.performance_log, exist_ok=True)
    devices = [int(i) for i in args.cuda_visible_devices.split(',')]
    print('Devices Available: ', devices)
    if len(devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=devices)
    model = model.cuda()
    print(model)
    train_dataset = FunnyNotFunnyDataset(root_dir=args.data_dir+'Train/', transform = transform, labels_path=args.data_dir+args.labels_path)
    valid_dataset = FunnyNotFunnyDataset(root_dir=args.data_dir+'Valid/', transform = transform, labels_path=args.data_dir+args.labels_path)
    test_dataset = FunnyNotFunnyDataset(root_dir=args.data_dir+'Test/' ,transform = transform, labels_path=args.data_dir+args.labels_path)
    
    labels = [item[1] for item in train_dataset.data]

    # Round the labels to nearest integer
    rounded_labels = [round(label) for label in labels]

    # Count occurrences of each label
    label_counts = torch.bincount(torch.tensor(rounded_labels).int())

    # Calculate weights
    weights = 1.0 / label_counts.float()
    sample_weights = weights[torch.tensor(rounded_labels).int()]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, sampler=sampler)
    valid_loader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True)
    valid_accuracy = train(num_epochs=args.num_epochs, save_name=args.output_file, model=model, train_dataloader=train_loader, valid_dataloader=valid_loader, LR=args.LR, weight_decay=args.weight_decay)
    model.load_state_dict(torch.load(args.output_file))
    test_accuracy = test(model=model, test_dataloader=test_loader)
    with open(os.path.join(args.performance_log, 'performance.txt'), 'a') as f:
        f.write('Experiment '+str(args.experiment_num)+ ': ' + f"test rmse: {test_accuracy} " + f"valid rmse: {valid_accuracy}" + '\n')