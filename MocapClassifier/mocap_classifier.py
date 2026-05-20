"""
Motion Capture Classification using LSTM Networks - Training Script

This module implements a neural network classifier for motion capture data using PyTorch.
The classifier processes windowed motion data from various file formats (BVH, FBX) and
trains an LSTM-based network to classify different motion types.
"""

"""
imports
"""

# General imports

import os
import sys
import time
import pickle
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

# Pytorch imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Mocap imports

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix

"""
Configurations
"""

# Device Settings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Mocap Settings

mocap_data_file_path = "E:/Data/mocap/Daniel/Zed/fbx/"
mocap_data_file_extensions = [".fbx"] 
mocap_joint_indices = [ 3, 4, 5, 6, 7 ] # right arm only
mocap_data_ids = ["rot_local"]
mocap_data_window_length = 90
mocap_data_window_offset = 15
mocap_pos_scale = 1.0

# Model settings

class_count = None # automatically determined 

model_input_dim = None # automatically determined
model_hidden_dim = 128
model_layer_count = 3
model_dropout = 0.3

load_weights = False
load_weights_epoch = 100

# Training settings

test_percentage = 0.2
batch_size = 64
epochs = 200
learning_rate = 1e-4
weight_decay = 0.001


"""
Load Data
"""

# verify that filename has allowed extension
def file_has_allowed_extension(filename, extensions):
    return filename.lower().endswith(tuple(extensions))

# find the class folders
def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

# generates a list of tuples (path_to_file, class_index)
def load_class_filenames(directory, extensions):

    _, class_to_idx = find_classes(directory)
    
    def is_valid_file(x: str) -> bool:
        return file_has_allowed_extension(x, extensions)
    
    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = directory + "/" + target_class

        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                
                path = root + "/" + fname
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

def load_mocap_file(mocap_file_path):
    
    print("process file path ", mocap_file_path)

    if mocap_file_path.endswith(".bvh") or mocap_file_path.endswith(".BVH"):
        bvh_data = bvh_tools.load(mocap_file_path)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    elif mocap_file_path.endswith(".fbx") or mocap_file_path.endswith(".FBX"):
        fbx_data = fbx_tools.load(mocap_file_path)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only
        
    mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    
    # set x and z offset of root joint to zero
    mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
    mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
    
    if mocap_file_path.endswith(".bvh") or mocap_file_path.endswith(".BVH"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    elif mocap_file_path.endswith(".fbx") or mocap_file_path.endswith(".FBX"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

    return mocap_data
    
# generates a list of tuples (data, class_index)
def load_class_data(class_files):
    
    class_data = []
    
    for class_file, class_index in class_files:
        
        #print("class_file: ", class_file, " class_index ", class_index)
        
        mocap_file_data = load_mocap_file(class_file)
        
        mocap_data = []
        
        for mocap_data_id in mocap_data_ids:
            
            data = mocap_file_data["motion"][mocap_data_id]
            
            #print("data 1 s ", data.shape)
            
            # filter joints
            data = data[:, mocap_joint_indices, :]
            
            #print("data 2 s ", data.shape)
            
            # combine joint count and joint dim into one dimension
            if len(data.shape) > 2:
                data = np.reshape(data, (data.shape[0], -1))
                
            #print("data 3 s ", data.shape)
            
            mocap_data.append(data)
            
            #print("mocap_data_id ", mocap_data_id, " data s ", data.shape)

        mocap_data = np.concatenate(mocap_data, axis=1)

        #print("mocap_data s ", mocap_data.shape)

        class_data.append( (mocap_data, class_index) )

    return class_data

        
classes, class_to_idx = find_classes(mocap_data_file_path)
class_files = load_class_filenames(mocap_data_file_path, mocap_data_file_extensions)
class_data = load_class_data(class_files)

"""
Create Dataset
"""

def create_dataset_with_split(class_data, window_length, window_offset, test_percentage):
    
    # Split files first, then create windows
    train_motion_data = []
    train_class_labels = []
    test_motion_data = []
    test_class_labels = []
    
    for data, class_id in class_data:
        data_count = data.shape[0]
        split_point = int((1 - test_percentage) * data_count)
        
        #print("class_id ", class_id, " data_count ", data_count, " split_point ", split_point)
        
        # Train data windows
        for dI in range(0, split_point - window_length, window_offset):
            motion_excerpt = data[dI:dI + window_length]
            train_class_labels.append(class_id)
            train_motion_data.append(motion_excerpt)
            
        # Test data windows
        for dI in range(split_point, data_count - window_length, window_offset):
            motion_excerpt = data[dI:dI + window_length]
            test_class_labels.append(class_id)
            test_motion_data.append(motion_excerpt)
    
    return np.array(train_class_labels, dtype=np.int64), np.stack(train_motion_data, dtype=np.float32), np.array(test_class_labels, dtype=np.int64), np.stack(test_motion_data, dtype=np.float32)

def check_class_distribution(train_class_labels, test_class_labels):
    # After creating your datasets, check class distribution
    print("Train class distribution:")
    unique, counts = np.unique(train_class_labels, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"Class {class_id} ({classes[class_id]}): {count} samples")
    
    print("\nTest class distribution:")
    unique, counts = np.unique(test_class_labels, return_counts=True)
    for class_id, count in zip(unique, counts):
        print(f"Class {class_id} ({classes[class_id]}): {count} samples")

def calculate_class_weights(class_labels, num_classes):
    """Calculate inverse frequency weights for classes"""
    unique, counts = np.unique(class_labels, return_counts=True)
    total_samples = len(class_labels)
    
    # Calculate weights inversely proportional to class frequency
    class_weights = np.zeros(num_classes)
    for class_id, count in zip(unique, counts):
        class_weights[class_id] = total_samples / (num_classes * count)
    
    return torch.FloatTensor(class_weights)

def calc_norm_values(train_motion_data):
    train_motion_data = np.reshape(train_motion_data, (-1, train_motion_data.shape[-1]))
    data_mean = np.mean(train_motion_data, axis=0)
    data_std = np.std(train_motion_data, axis=0)
    
    # Add small epsilon to prevent division by zero
    eps = 1e-8
    data_std = np.maximum(data_std, eps)
    
    return data_mean, data_std


class MotionDataset(Dataset):
    
    def __init__(self, class_labels, motion_data):
        
        self.class_labels = class_labels
        self.motion_data = motion_data
        
    def __getitem__(self, index):

        x = self.motion_data[index]
        y = self.class_labels[index]
        
        return x, y

    def __len__(self):
        return len(self.class_labels)    
       
train_class_labels, train_motion_data, test_class_labels, test_motion_data = create_dataset_with_split(class_data, mocap_data_window_length, mocap_data_window_offset, test_percentage)
check_class_distribution(train_class_labels, test_class_labels)
data_mean, data_std = calc_norm_values(train_motion_data)
class_weights = calculate_class_weights(train_class_labels, len(classes))
print("Class weights:", class_weights)

# save data mean and std
with open("results/data/mean.pkl", 'wb') as f:
    pickle.dump(data_mean, f)
with open("results/data/std.pkl", 'wb') as f:
    pickle.dump(data_std, f)


train_dataset = MotionDataset(train_class_labels, train_motion_data)
test_dataset = MotionDataset(test_class_labels, test_motion_data)

item_x, item_y = train_dataset[0]
print("item_x s ", item_x.shape)
print("item_y s ", item_y.shape)


# create dataloader
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

batch_x, batch_y = next(iter(trainloader))

print("batch_x s ", batch_x.shape)
print("batch_y s ", batch_y.shape)

"""
Create Model
"""

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_count, class_count, dropout):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_count, batch_first=True, dropout=0.0)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim // 2, class_count)
        self.relu = nn.ReLU()
        
        self.init_weights(self.rnn)
        self.init_weights(self.fc1)
        self.init_weights(self.fc2)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.orthogonal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)

    def forward(self, x):
        x, (h, c) = self.rnn(x)
        x = h[-1]  # Take final hidden state from last layer
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    
model_input_dim = item_x.shape[-1]
class_count = len(classes)

classifier = Classifier(model_input_dim, model_hidden_dim, model_layer_count, class_count, model_dropout)
classifier.to(device)

print(classifier)

# test classifier

batch = next(iter(trainloader))

batch_x = batch[0].to(device)
batch_y = batch[1].to(device)

print("batch_x s ", batch_x.shape)
print("batch_y s ", batch_y.shape)

batch_yhat = classifier(batch_x)

print("batch_yhat s ", batch_yhat.shape)

if load_weights and load_weights_epoch > 0:
    classifier.load_state_dict(torch.load("results/weights/classifier_epoch_{}".format(load_weights_epoch)))

"""
Training
"""

class_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))

optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Lower learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)  # Less aggressive decay

data_mean = torch.tensor(data_mean, dtype=torch.float32).reshape(1, 1, -1).to(device)
data_std = torch.tensor(data_std, dtype=torch.float32).reshape(1, 1, -1).to(device)


def train_step(batch_x, batch_y):
    
    batch_x_norm = (batch_x - data_mean) / data_std 
    batch_x_norm = torch.nan_to_num(batch_x_norm)

    batch_yhat = classifier(batch_x_norm)
    _loss = class_loss(batch_yhat, batch_y) 

    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    
    return _loss

def test_step(batch_x, batch_y):
    
    batch_x_norm = (batch_x - data_mean) / (data_std + 1e-8)

    with torch.no_grad():
        batch_yhat = classifier(batch_x_norm)
        _loss = class_loss(batch_yhat, batch_y) 
        
    return _loss

def test_model(data_loader):
    
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            
            batch_x, batch_y = data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            batch_x_norm = (batch_x - data_mean) / (data_std + 1e-8)
        
            outputs = classifier(batch_x_norm)
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        return correct / total

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["train"] = []
    loss_history["test"] = []
    
    for epoch in range(epochs):
        
        start = time.time()
        
        train_loss_per_epoch = []
        
        for train_data in train_dataloader:
            
            batch_x, batch_y = train_data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            _train_loss = train_step(batch_x, batch_y)
            _train_loss = _train_loss.detach().cpu().numpy()
            train_loss_per_epoch.append(_train_loss)
            
        train_loss_per_epoch = np.mean(np.array(train_loss_per_epoch))
        
        loss_history["train"].append(train_loss_per_epoch)
        
        
        test_loss_per_epoch = []
        
        for test_data in test_dataloader:
            
            batch_x, batch_y = test_data
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            _test_loss = test_step(batch_x, batch_y)
            _test_loss = _test_loss.detach().cpu().numpy()
            test_loss_per_epoch.append(_test_loss)
            
        test_loss_per_epoch = np.mean(np.array(test_loss_per_epoch))
        
        loss_history["test"].append(test_loss_per_epoch)
        
        
        train_correct = test_model(train_dataloader)
        test_correct = test_model(test_dataloader)
        
        scheduler.step()
        
        print ('epoch {} : train: loss {:01.4f} corr {:01.2f} test: loss {:01.4f} correct {:01.2f} time {:01.2f}'.format(epoch + 1, train_loss_per_epoch, train_correct * 100, test_loss_per_epoch, test_correct * 100, time.time()-start))

    return loss_history

# fit model
loss_history = train(trainloader, testloader, epochs)

"""
Save Training
"""

def save_loss_as_csv(loss_history, csv_file_name):
    with open(csv_file_name, 'w') as csv_file:
        csv_columns = list(loss_history.keys())
        csv_row_count = len(loss_history[csv_columns[0]])
        
        
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=',', lineterminator='\n')
        csv_writer.writeheader()
    
        for row in range(csv_row_count):
        
            csv_row = {}
        
            for key in loss_history.keys():
                csv_row[key] = loss_history[key][row]

            csv_writer.writerow(csv_row)


def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        plt.plot(range(epochs), loss_history[key], label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_file_name)
    plt.show()
    
save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save trained model
PATH = 'results/weights/classifier_weights_epoch_{}.pth'.format(epochs)
torch.save(classifier.state_dict(), PATH)

# Test the model on some test data
batch_x, batch_y = next(iter(testloader))
batch_x = batch_x.to(device)
batch_x_norm = (batch_x - data_mean) / data_std 
batch_x_norm = torch.nan_to_num(batch_x_norm)
batch_yhat = classifier(batch_x_norm)
_, pred_labels = torch.max(batch_yhat, 1)

for i in range(batch_size):
    print("motion {} pred class {} true class {}".format(i, pred_labels[i], batch_y[i]))
    

