"""
Motion Capture Classification using Structure-Aware LSTM
"""

"""
Imports
"""

import os, time
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Mocap imports
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap

"""
Settings
"""

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
Mocap Settings
"""

mocap_data_file_path = "E:/Data/mocap/Daniel/Zed/fbx/classes/"
mocap_data_file_extensions = [".fbx"] 
mocap_joint_indices = [3, 4, 5, 6, 7] # right arm only
mocap_data_window_length = 90
mocap_data_window_offset = 15
mocap_pos_scale = 1.0
mocap_stats_load = False # load previously calculated mocap statistic instead of calculating a new one

"""
Model Settings
"""

class_count = None # will be calculated

model_input_dim = None # will be calculated
model_hidden_dim = 128
model_layer_count = 2
model_dropout = 0.3

"""
Training Settings
"""

test_percentage = 0.2
batch_size = 64
epochs = 200
learning_rate = 1e-4
label_smoothing = 0.1
load_weights = False
save_weights = True
model_weights_file = "results_v2/weighs/classifier_epoch_200.pth"

"""
Save Paths Settings
"""

save_path = "results_v2"
save_stats_path = save_path + "/stats"
save_history_path = save_path + "/history"
save_weights_path = save_path + "/weights"

os.makedirs(save_stats_path, exist_ok=True)
os.makedirs(save_history_path, exist_ok=True)
os.makedirs(save_weights_path, exist_ok=True)

"""
Load Class and Mocap Data
"""

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

def load_class_filenames(directory, extensions):
    _, class_to_idx = find_classes(directory)
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir): continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if fname.lower().endswith(tuple(extensions)):
                    instances.append((os.path.join(root, fname), class_index))
    return instances

def load_mocap_file(mocap_file_path):
    if mocap_file_path.lower().endswith(".bvh"):
        bvh_data = bvh_tools.load(mocap_file_path)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    elif mocap_file_path.lower().endswith(".fbx"):
        fbx_data = fbx_tools.load(mocap_file_path)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0]
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])

    mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
    mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
    return mocap_data

def load_class_data(class_files):
    class_data = []
    for class_file, class_index in class_files:
        mocap_file_data = load_mocap_file(class_file)
        
        pos = mocap_file_data["motion"]["pos_local"][:, mocap_joint_indices, :]
        rot = mocap_file_data["motion"]["rot_local"][:, mocap_joint_indices, :]
        
        # Calculate First Derivative: Velocity
        vel_pos = np.concatenate((np.zeros((1, pos.shape[1], pos.shape[2])), np.diff(pos, axis=0)), axis=0)
        vel_rot = np.concatenate((np.zeros((1, rot.shape[1], rot.shape[2])), np.diff(rot, axis=0)), axis=0)
        
        # Calculate Second Derivative: Acceleration
        acc_pos = np.concatenate((np.zeros((1, vel_pos.shape[1], vel_pos.shape[2])), np.diff(vel_pos, axis=0)), axis=0)
        acc_rot = np.concatenate((np.zeros((1, vel_rot.shape[1], vel_rot.shape[2])), np.diff(vel_rot, axis=0)), axis=0)
        
        # Final shape: (frames, num_joints, 21 features)
        mocap_data = np.concatenate([pos, rot, vel_pos, vel_rot, acc_pos, acc_rot], axis=-1)
        class_data.append((mocap_data, class_index))
    return class_data

def calculate_class_weights(class_labels, num_classes):
    """Calculate inverse frequency weights for classes"""
    unique, counts = np.unique(class_labels, return_counts=True)
    total_samples = len(class_labels)

    # Calculate weights inversely proportional to class frequency
    class_weights = np.zeros(num_classes)
    for class_id, count in zip(unique, counts):
        class_weights[class_id] = total_samples / (num_classes * count)

    return torch.FloatTensor(class_weights)


"""
Normalise Data and Create Dataset
"""

def create_dataset_with_split(class_data, window_length, window_offset, test_percentage):
    train_motion, train_labels, test_motion, test_labels = [], [], [], []
    for data, class_id in class_data:
        split_point = int((1 - test_percentage) * data.shape[0])
        for dI in range(0, split_point - window_length, window_offset):
            train_labels.append(class_id)
            train_motion.append(data[dI:dI + window_length])
        for dI in range(split_point, data.shape[0] - window_length, window_offset):
            test_labels.append(class_id)
            test_motion.append(data[dI:dI + window_length])
    return np.array(train_labels, dtype=np.int64), np.stack(train_motion, dtype=np.float32), \
           np.array(test_labels, dtype=np.int64), np.stack(test_motion, dtype=np.float32)

def calc_norm_values(train_motion_data):
    flattened = np.reshape(train_motion_data, (-1, train_motion_data.shape[-1]))
    mean = np.mean(flattened, axis=0)
    std = np.std(flattened, axis=0)
    std[std == 0] = 1e-8
    return mean, std

class MocapDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

class Classifier(nn.Module):
    def __init__(self, num_features=21, hidden_dim=128, num_classes=5, num_layers=2, dropout=0.3):
        super().__init__()
        self.joint_embedding = nn.Linear(num_features, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, num_joints, num_features)
        x = self.joint_embedding(x)
        x, _ = torch.max(x, dim=2) # Spatial Max-Pooling across joints
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]) # Classify based on final time step
    
"""
Training Functions
"""

def train_step(batch_x, batch_y):
    batch_x_norm = (batch_x - data_mean) / (data_std + 1e-8)
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

    classifier.eval() # Ensure dropout is disabled for eval
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
        classifier.train() # Enable dropout during training
        
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
        classifier.eval() # Disable dropout during test loss calculation
        
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
        
        print ('epoch {} : train: loss {:01.4f} corr {:01.2f} test: loss {:01.4f} correct {:01.2f} time {:01.2f}'.format(
            epoch + 1, train_loss_per_epoch, train_correct * 100, test_loss_per_epoch, test_correct * 100, time.time()-start))

    return loss_history

"""
Save Training History
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

"""
Main Execution
"""

# detect classes and load corresponding motion capture files

classes, class_to_idx = find_classes(mocap_data_file_path)
class_count = len(classes)
class_files = load_class_filenames(mocap_data_file_path, mocap_data_file_extensions)
class_data = load_class_data(class_files)
    
# create dataset 
train_labels, train_data, test_labels, test_data = create_dataset_with_split(
    class_data, mocap_data_window_length, mocap_data_window_offset, test_percentage)

model_input_dim = train_data.shape[-1]

# normalise motion data
if mocap_stats_load == False:
    
    mean_np, std_np  = calc_norm_values(train_data)
    
    np.save(save_stats_path + "/mean.npy", mean_np)
    np.save(save_stats_path + "/std.npy", std_np)

else:
    
    mean_np = np.load(save_stats_path + "/mean.npy")
    std_np = np.load(save_stats_path + "/std.npy")
    
data_mean = torch.tensor(mean_np, dtype=torch.float32).to(device)
data_std = torch.tensor(std_np, dtype=torch.float32).to(device)

# Calculate class weights for unbalanced datasets
class_weights = calculate_class_weights(train_labels, len(classes)).to(device)

# create data loaders
train_loader = DataLoader(MocapDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(MocapDataset(test_data, test_labels), batch_size=batch_size, shuffle=True)

# create classifier model
classifier = Classifier(num_features=model_input_dim, hidden_dim=model_hidden_dim, num_classes=class_count, num_layers=model_layer_count, dropout=model_dropout).to(device)

print(classifier)

# test classifier

batch = next(iter(train_loader))

batch_x = batch[0].to(device)
batch_y = batch[1].to(device)

print("batch_x s ", batch_x.shape)
print("batch_y s ", batch_y.shape)

batch_yhat = classifier(batch_x)

print("batch_yhat s ", batch_yhat.shape)

if load_weights:
    classifier.load_state_dict(torch.load(model_weights_file))

# run training

if save_weights:
    
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)  # Less aggressive decay
    class_loss = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=label_smoothing)

    print("Starting Training...")
    history = train(train_loader, test_loader, epochs)
    
    save_loss_as_csv(history, save_history_path + "/history_{}.csv".format(epochs))
    save_loss_as_image(history, save_history_path + "/history_{}.png".format(epochs))

    # save trained model
    torch.save(classifier.state_dict(), save_weights_path + "/classifier_weights_epoch_{}.pth".format(epochs))


# Test the model on some test data
batch_x, batch_y = next(iter(test_loader))
batch_x = batch_x.to(device)
batch_x_norm = (batch_x - data_mean) / data_std 
batch_x_norm = torch.nan_to_num(batch_x_norm)
batch_yhat = classifier(batch_x_norm)
_, pred_labels = torch.max(batch_yhat, 1)

for i in range(batch_size):
    print("motion {} pred class {} true class {}".format(i, pred_labels[i], batch_y[i]))