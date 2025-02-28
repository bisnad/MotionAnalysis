"""
imports
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as nnF
import torch.optim as optim
from collections import OrderedDict

import os, sys, time, subprocess
import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt


"""
Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Dataset
"""

data_file_path = "../../AIToolbox/Data/Mocap/Stocos/Solos/MovementQualities_IMU/"
data_file_extensions = ["pkl"] 
data_sensor_ids = ["/accelerometer", "/gyroscope"]
data_window_length = 60
data_window_offset = 1

"""
Model settings
"""

input_dim = None # automatically determined
hidden_dim = 32
layer_count = 3
class_count = None # automatically determined 

"""
Training settings
"""

test_percentage = 0.2
batch_size = 32
learning_rate = 1e-3
epochs = 100

load_weights = False
load_weights_epoch = 100

"""
Load Data
"""

# verify that filename has allowed extension
def file_has_allowed_extension(filename, extensions):
    return filename.lower().endswith(tuple(extensions))

# load all recording files into a dictionary
def load_recordings(data_file_path, extensions):
    
    files_data = []
    
    for root, _, file_names in sorted(os.walk(data_file_path, followlinks=True)):
        for file_name in sorted(file_names):
            file_path = root + "/" + file_name
                        
            #print("file_path: ", file_path)
            
            with open(file_path, "rb") as input_file:
                file_data = pickle.load(input_file)
                
                files_data.append(file_data)

    return files_data

# detect numner of classes
def get_class_count(recording_files):
    return len(set([ recording_file["class_id"] for recording_file in recording_files ]))

# filter and concatenate sensor values
def process_recordings(data_files, data_sensor_ids):
    
    recordings = []
    
    for data_file in data_files:
        
        recording = {}
        recording["class_id"] = data_file["class_id"]
        
        sensor_values_combined = []
        
        for sensor_id in data_sensor_ids:
            
            #print("sensor_id ", sensor_id)

            sensor_indices = [i for i, x in enumerate(data_file["sensor_ids"]) if x == sensor_id]
            sensor_values = [data_file["sensor_values"][i] for i in sensor_indices]
            sensor_values = np.array(sensor_values)
            
            #print("sensor_values s ", sensor_values.shape)
        
            sensor_values_combined.append(sensor_values)
            
        recording["sensor_values"] = np.concatenate(sensor_values_combined, axis=1)

        recordings.append(recording)
                  
    return recordings

# calculate mean and std of sensor value
def calc_norm_values(recording_data):
    
    all_values = []
    
    for data in recording_data:
        all_values.append(data["sensor_values"])
    
    all_values = np.concatenate(all_values, axis=0)
    
    values_mean = np.mean(all_values, axis=0)
    values_std = np.std(all_values, axis=0)
    
    return values_mean, values_std

# create dataset
def create_dataset(recording_data, window_length, window_offset):
    
    sensor_data = []
    class_labels = []
    
    for recording in recording_data:
        
        class_id = recording["class_id"]
        sensor_values = recording["sensor_values"]
        sensor_value_count = sensor_values.shape[0]
        
        for eI in range(0, sensor_value_count - window_length, window_offset):
            sensor_values_excerpt = sensor_values[eI:eI + window_length]
            
            class_labels.append(class_id)
            sensor_data.append(sensor_values_excerpt)
            
    class_labels = np.array(class_labels, dtype=np.int64)
    sensor_data = np.stack(sensor_data, axis=0).astype(np.float32)
        
    return class_labels, sensor_data
                

"""
Load and Process Sensor Data Recordings
"""

recording_files = load_recordings(data_file_path, data_file_extensions)
class_count = get_class_count(recording_files)
recording_data = process_recordings(recording_files, data_sensor_ids)     

"""
Compute Sensor Data Mean and Std
"""

data_mean, data_std = calc_norm_values(recording_data)

# save data mean and std
with open("results/data/mean.pkl", 'wb') as f:
    pickle.dump(data_mean, f)
with open("results/data/std.pkl", 'wb') as f:
    pickle.dump(data_std, f)

"""
Create Dataset
"""

class_labels, sensor_data = create_dataset(recording_data, data_window_length, data_window_offset)

class SensorDataset(Dataset):
    
    def __init__(self, class_labels, sensor_data):
        
        self.class_labels = class_labels
        self.sensor_data = sensor_data
        
    def __getitem__(self, index):

        x = self.sensor_data[index]
        y = self.class_labels[index]
        
        return x, y

    def __len__(self):
        return len(self.class_labels)    
        

dataset = SensorDataset(class_labels, sensor_data)

item_x, item_y = dataset[0]
print("item_x s ", item_x.shape)
print("item_y s ", item_y.shape)

# train test split
dataset_size = len(dataset)
test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# create dataloader
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

batch_x, batch_y = next(iter(trainloader))

print("batch_x s ", batch_x.shape)
print("batch_y s ", batch_y.shape)

"""
Create Model
"""

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_count, class_count):
        super().__init__()

        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_count, batch_first=True)
        self.fc = nn.Linear(hidden_dim, class_count)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x):

        x, (h, c) = self.rnn(x)
        x = x[:, -1, :] # only last time step 
        x = self.fc(x)
        y = self.sm(x)
        return y
    
input_dim = item_x.shape[-1]
    
classifier = Classifier(input_dim, hidden_dim, layer_count, class_count)
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
    classifier.load_state_dict(torch.load("results/weights/classifier_epoch_{}.pth".format(load_weights_epoch)))

"""
Training
"""

class_loss = nn.NLLLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # reduce the learning every 20 epochs by a factor of 10

data_mean = torch.tensor(data_mean, dtype=torch.float32).reshape(1, 1, -1).to(device)
data_std = torch.tensor(data_std, dtype=torch.float32).reshape(1, 1, -1).to(device)

def train_step(batch_x, batch_y):
    
    batch_x_norm = (batch_x - data_mean) / data_std 
    
    batch_yhat = classifier(batch_x_norm)
    _loss = class_loss(batch_yhat, batch_y) 

    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    
    return _loss

def test_step(batch_x, batch_y):
    
    batch_x_norm = (batch_x - data_mean) / data_std 

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
            
            batch_x_norm = (batch_x - data_mean) / data_std 
        
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
batch_yhat = classifier(batch_x.to(device))
_, pred_labels = torch.max(batch_yhat, 1)

for i in range(batch_size):
    print("motion {} pred class {} true class {}".format(i, pred_labels[i], batch_y[i]))
