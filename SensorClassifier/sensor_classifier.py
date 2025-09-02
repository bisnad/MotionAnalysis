"""
Sensor Classification using LSTM Networks

This module implements a neural network classifier for sensor data using PyTorch.
The classifier processes windowed motion data and
trains an LSTM-based network to classify different motion types.
"""

"""
imports
"""

# General imports
import os, sys, time, subprocess
import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt
import pathlib
from collections import OrderedDict

# Pytorch imports

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as nnF
import torch.optim as optim

"""
Configuration
"""

# Device settings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Sensor data settings

sensor_data_file_path = "data/sensors/"
sensor_data_file_extensions = [".pkl"] 
#sensor_data_ids = ["/accelerometer", "/gyroscope"]
sensor_data_ids = ["/accelerometer"]
sensor_data_window_length = 90
#sensor_data_window_offset = 15
sensor_data_window_offset = 5

# Model settings

class_count = None # automatically determined 

model_input_dim = None # automatically determined
model_hidden_dim = 128
model_layer_count = 3
model_dropout = 0.3

load_weights = False
load_weights_epoch = 100

"""
Training settings
"""

test_percentage = 0.2
batch_size = 64
epochs = 200
learning_rate = 1e-4
weight_decay = 0.001

load_weights = False
load_weights_epoch = 100

"""
Create Results Directories
"""

os.makedirs("results/data", exist_ok=True)
os.makedirs("results/histories", exist_ok=True)
os.makedirs("results/weights", exist_ok=True)

"""
Load Data
"""

"""Check if filename has one of the allowed extensions."""
def file_has_allowed_extension(filename, extensions):
    return filename.lower().endswith(tuple(extensions))

"""Find class folders and create class-to-index mapping."""
def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

"""Generate list of tuples (path_to_file, class_index)."""
def load_class_filenames(directory, extensions):

    _, class_to_idx = find_classes(directory)
        
    instances = []
    available_classes = set()
    
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        
        if not os.path.isdir(target_dir):
            continue
            
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if file_has_allowed_extension(path, extensions):
                    instances.append((path, class_index))
                    available_classes.add(target_class)
    
    # Check for empty classes
    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        extensions_str = ', '.join(extensions) if isinstance(extensions, list) else extensions
        raise FileNotFoundError(
            f"Found no valid files for classes {', '.join(sorted(empty_classes))}. "
            f"Supported extensions: {extensions_str}"
        )
    
    return instances

def load_sensor_recording(file_path):
    
    print(f"Processing file: {file_path}")
        
    file_ext = file_path.lower()
    
    with open(file_path, "rb") as input_file:
        sensor_data = pickle.load(input_file)
        
    return sensor_data

def load_sensor_recordings(class_files):
    
    recording_data = []
    
    for file_path, class_index in class_files:
        file_data = load_sensor_recording(file_path)
        
        recording_data.append(file_data)
        
    return recording_data

# filter and concatenate sensor values
def process_sensor_recordings(recording_data, sensor_data_ids):
    
    class_data = []
    
    for file_data in recording_data:
        
        class_id = file_data["class_id"]
        
        min_data_count = None
        
        sensor_values_combined = []
        
        for sensor_id in sensor_data_ids:
            
            print("sensor_id ", sensor_id)

            sensor_indices = [i for i, x in enumerate(file_data["sensor_ids"]) if x == sensor_id]
            sensor_values = [file_data["sensor_values"][i] for i in sensor_indices]
            sensor_values = np.array(sensor_values)
            
            print("sensor_values s ", sensor_values.shape)
            
            if min_data_count is None:
                min_data_count = sensor_values.shape[0]
            elif min_data_count > sensor_values.shape[0]:
                min_data_count = sensor_values.shape[0]
                
            sensor_values = sensor_values[:min_data_count, ...]
            
            #print("sensor_values2 s ", sensor_values.shape)
        
            sensor_values_combined.append(sensor_values)
            
        sensor_values_combined = np.concatenate(sensor_values_combined, axis=1)
        
        class_data.append( (sensor_values_combined, class_id) )
                  
    return class_data

def load_class_data(class_files, sensor_data_ids):
    recording_data = load_sensor_recordings(class_files)
    class_data = process_sensor_recordings(recording_data, sensor_data_ids)

    return class_data

"""
Load and Process Sensor Data Recordings
"""

classes, class_to_idx = find_classes(sensor_data_file_path)
class_files = load_class_filenames(sensor_data_file_path, sensor_data_file_extensions)
class_data = load_class_data(class_files, sensor_data_ids)

"""
Create Dataset
"""

# create train and test data split
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
    """PyTorch Dataset for motion capture data."""
    
    def __init__(self, class_labels: np.ndarray, motion_data: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            class_labels: Array of class indices
            motion_data: Array of motion sequences
        """
        self.class_labels = class_labels
        self.motion_data = motion_data
    
    def __getitem__(self, index):
        """Get a single item from the dataset."""
        return self.motion_data[index], self.class_labels[index]
    
    def __len__(self):
        """Return dataset length."""
        return len(self.class_labels)
        
train_class_labels, train_motion_data, test_class_labels, test_motion_data = create_dataset_with_split(class_data, sensor_data_window_length, sensor_data_window_offset, test_percentage)
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
    
