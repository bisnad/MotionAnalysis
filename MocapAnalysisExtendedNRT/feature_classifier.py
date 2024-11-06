import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm

from collections import OrderedDict

import os, sys, time, subprocess
import numpy as np
import random
import json
import pickle
from tqdm import trange, tqdm
import shutil
import re
import glob



"""
Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


"""
Settings: Motion Features, Motion Labels
"""

analysis_window_size = 16 # should be the same value used for motion feature analysis
features_file_path = "results/motion_features/"
labels_file_path = "results/motion_labels/"

feature_files = ["Canal_14-08-2023-001_Muriel_Fluid_1_features.npy",
                 "Canal_14-08-2023-004_Muriel_staccato_full_body_features.npy"]

label_files = ["Canal_14-08-2023-001_Muriel_Fluid_1_labels.json",
               "Canal_14-08-2023-004_Muriel_staccato_full_body_labels.json"]

"""
Load Motion Features
"""

motion_features = []

for feature_file in feature_files:
    features = np.load(features_file_path + feature_file, allow_pickle=True)
    motion_features.append(features)
    
motion_feature = motion_features[0]

# TODO. figure out how to get the stored dictionaries back from the numpy object

type(motion_feature)

motion_feature_dict = dict(motion_feature)

# TOOD: convert numpy object into dictionary


bvh_tools = bvh.BVH_Tools()
mocap_tools = mocap.Mocap_Tools()

def load_body_parts(path, file):
    with open(path + file) as file:
        body_parts = json.load(file)
    
    return body_parts["body_parts"]

def load_mocap_data(path, files):
    all_mocap_data = []
    
    for file in files:
        
        print("file ", file)
        
        file_wo_suffix = os.path.splitext(file)[0]
        file_proc_path = path + file_wo_suffix + ".p"
        
        if os.path.isfile(file_proc_path) == True:
            mocap_data = pickle.load(open(file_proc_path, "rb"))
        else:
            bvh_data = bvh_tools.load(path + file)
            mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
            mocap_data = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
            
            pickle.dump(mocap_data, open(file_proc_path, "wb"))
        
        all_mocap_data.append(mocap_data)

    #mocap_data = np.concatenate(mocap_data, axis=0)
    
    #print("total shape ", mocap_data.shape)

    return all_mocap_data

def load_label_data(body_parts, path, files):
    all_data = []
    for file in files:
        part_dict = {}
        
        with open(path + file) as json_file:
            label_dict = json.load(json_file)
            
            for part_label, frame_ranges in label_dict.items():
                label, bodypart = part_label.split(" ")
                
                if bodypart in body_parts == False:
                    continue
                
                #print("bodypart ", bodypart, " label ", label, " frames ", frame_ranges)
                
                if bodypart not in part_dict:
                    part_dict[bodypart] = {}
                
                if label not in part_dict[bodypart]:
                    part_dict[bodypart][label] = []
                
                for frame_range in frame_ranges:
                    part_dict[bodypart][label].append(frame_range)
                
        all_data.append(part_dict)  

    return all_data

body_parts = load_body_parts(mocap_file_path, body_parts_file)
mocap_data = load_mocap_data(mocap_file_path, mocap_files)
label_data = load_label_data(body_parts, labels_file_path, label_files)

#Simplify data since we are dealing with RightArm only (so no distinction between body parts necessary)
label_data = [ labels["RightArm"] for labels in label_data ]
mocap_data = [ mocap[:, np.array(body_parts["RightArm"]), :] for mocap in mocap_data ]

class MotionDataset(Dataset):
    def __init__(self, label_data, mocap_data, excerpt_size, offset_size, offset_type='random'):
        
        self.excerpt_size = excerpt_size
        self.offset_size = offset_size
        self.offset_type = offset_type
        
        # gather quality names and indices
        self.quality_names = list(set([ label for labels in label_data for label in labels ]))
        self.quality_names.sort()
        self.quality_names_to_indices = {name: index for index, name in enumerate(self.quality_names)}
        self.quality_indices_to_names = {index: name for index, name in enumerate(self.quality_names)}
        
        self.label_data = []
        self.mocap_data = []
        
        for idx, (labels, mocap) in enumerate(zip(label_data, mocap_data)):
            
            #print("idx ", idx)
            
            for label, frame_ranges in labels.items():
                    
                #print("label ", label)
                
                label_idx = self.quality_names_to_indices[label]
                
                for frame_range in frame_ranges:
                    
                    #print("frame_range ", frame_range)
                    
                    range_begin = frame_range[0]
                    range_end = frame_range[1]
                    
                    for frame in range(range_begin, range_end - excerpt_size - offset_size, offset_size):
                        
                        excerpt_begin = frame
                        excerpt_end = excerpt_begin + excerpt_size + offset_size
                        
                        #print("eb ", excerpt_begin, " ee ", excerpt_end)
                        
                        self.label_data.append(label_idx)
                        self.mocap_data.append(mocap[excerpt_begin:excerpt_end, ...])
        
    def quality_count(self):
        return len(self.quality_names)
        
    def joint_count(self):
        return self.mocap_data[0].shape[1]

    
    def joint_dim(self):
        return self.mocap_data[0].shape[2]
        
    def quality_name_2_idx(self, name):
        return self.quality_names_to_indices[name]
    
    def quality_idx_2_name(self, idx):
        return self.quality_indices_to_names[idx]
    
    def __len__(self):
        return len(self.mocap_data)
    
    def __getitem__(self, index):

        mocap_item = self.mocap_data[index]
        label_item = self.label_data[index]
        
        if self.offset_type == "random":
            start_frame = random.randint(0, self.offset_size - 1)
        else:
            start_frame = 0
        
        end_frame = start_frame + self.excerpt_size
        mocap_item = torch.from_numpy(mocap_item[start_frame:end_frame]).to(torch.float32)
        
        label_item = torch.Tensor([label_item]).to(torch.int32)
        
        return mocap_item, label_item

        
mocap_excerpt_size = 50
mocap_excerpt_offset = 10
mocap_excerpt_offset_type = "random"
            
full_dataset = MotionDataset(label_data, mocap_data, mocap_excerpt_size, mocap_excerpt_offset, mocap_excerpt_offset_type)    

data_x, data_y = full_dataset[0]

print("data_x s ", data_x.shape)
print("data_y s ", data_y.shape)

test_percentage = 0.1
batch_size = 1

dataset_size = len(full_dataset)
test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

batch_x, batch_y = next(iter(train_loader))
print("batch_x s ", batch_x.shape)
print("batch_y s ", batch_y.shape)


# store quality names to indices dict for use in inference only implementations

label_index_file = "MUR_Label_Indices.json"
json.dump(full_dataset.quality_indices_to_names, open(labels_file_path + "/" + label_index_file, "w"))

"""
Model
"""

pose_dim = full_dataset.joint_count() * full_dataset.joint_dim()
quality_count = full_dataset.quality_count()

model_input_dim = pose_dim
model_output_dim = quality_count
model_embed = 48
model_rnn_bidirectional = True
model_rnn_hidden_dim = 1024
model_rmm_layer_count = 2
model_dense_layer_count = 1
model_dropout = 0.5

class Classifier(nn.Module):
    def __init__(self, in_size, out_size, hidden=128, dropout=0.5, bidirectional=True, stack=1, layers=1, embed=0):
        super(Classifier, self).__init__()
        self.in_size = in_size
        self.bidirectional = bidirectional
        rnn_hidden = hidden // 2 if bidirectional else hidden

        self.embed = nn.Sequential(
                        nn.Linear(in_size, embed),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                     ) if embed > 0 else None

        self.lstm = nn.LSTM(embed if embed > 0 else in_size, rnn_hidden,
                            num_layers=stack,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        classifier_layers = []
        for _ in range(layers - 1):
            classifier_layers.append(nn.Linear(hidden, hidden))
            classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Dropout(dropout))
        classifier_layers.append(nn.Linear(hidden, out_size))
        classifier_layers.append(nn.LogSoftmax(dim=1))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward_lstm(self, input):
        input = self.embed(input) if self.embed is not None else input  # embed all data in the sequence
        return self.lstm(input)

    def forward(self, input):
        
        #print("input s ", input.shape)
        
        outputs, hidden = self.forward_lstm(input)
        
        #print("outputs s ", outputs.shape)
        
        last_out = outputs[:, -1, :]  # this is the last hidden state (last timestep) of the last stacked layer
        
        #print("last_out s ", last_out.shape)
        
        yhat = self.classifier(last_out)
        
        #print("yhat s ", yhat.shape)
        
        return yhat

    def segment(self, input):
        outputs, hidden = self.forward_lstm(input)
        return self.classifier(outputs)

    def steps_forward(self, input):
        outputs, hidden = self.forward_lstm(input)

        ''' for bidirectional models, we have to reverse the hidden states
            of the second direction in order to have the combined hidden state
            for each time step
        '''
        if self.bidirectional:
            seq_len = input.shape[0]
            outputs = outputs.view(seq_len, 2, -1)
            idx = torch.LongTensor([i for i in range(seq_len - 1, -1, -1)])
            if outputs.is_cuda:
                idx = idx.cuda()
            idx = Variable(idx, requires_grad=False)
            outputs[:, 1] = outputs[:, 1].index_select(0, idx)
            outputs = outputs.view(seq_len, -1)

        return self.classifier(outputs)

    def extract(self, input):
        outputs, hidden = self.forward_lstm(input)
        last_out = hidden[1].view(1, -1)
        return last_out

classifier = Classifier(model_input_dim, model_output_dim, model_rnn_hidden_dim, model_dropout, model_rnn_bidirectional, model_rmm_layer_count, model_dense_layer_count,model_embed).to(device)

print(classifier)

batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[1], pose_dim).to(device)
batch_yhat = classifier(batch_x)

_, pred_labels = torch.max(batch_yhat, 1)
true_action_idx = batch_y.detach().cpu().item()
pred_action_idx = pred_labels.detach().cpu().item()

true_action_idx
pred_action_idx

"""
Training
"""

epochs = 150
learning_rate = 0.0005
weight_decay = 1e-4
label_smoothing = 0.1
accumulate = 40
clip_norm = False
log_every = accumulate
log_file = os.path.join('log.txt')
log = open(log_file, 'a+')
no_progress = False

optimizer = Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    output = output.data if isinstance(output, Variable) else output

    vals, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k.mul_(100.0 / batch_size)
        res.append(correct_k[0])
    return res

def compute_loss(output, target):

    n_classes = output.shape[1]

    # build one-hot vector
    target = target.data.cpu()[0]
    y = torch.zeros(1, n_classes)
    y[0, target] = 1

    y = y.to(device)
    y = Variable(y, requires_grad=False)

    if label_smoothing:
        y = y * (1 - label_smoothing) + \
            (1 - y) * label_smoothing / (n_classes - 1)

    return F.binary_cross_entropy_with_logits(output, y)

def evaluate(loader, model):
    model.eval()

    avg_loss = 0.0
    avg_acc1 = 0.0
    avg_acc5 = 0.0

    n_classes = loader.dataset.dataset.quality_count()
    action_correct = torch.zeros(n_classes)
    action_count = torch.zeros(n_classes)

    progress_bar = tqdm(loader, disable=no_progress, position=0, leave=True)
    for i, (x, y) in enumerate(progress_bar):
        _y = y
        
        x = x.reshape(x.shape[0], x.shape[1], pose_dim).to(device)

        x = x.to(device)
        y = y.to(device)

        x = Variable(x, volatile=True)
        y = Variable(y, volatile=True)

        y_hat = model(x)

        loss = compute_loss(y_hat, y)
        
        avg_loss += loss.item()

        acc1, acc5 = accuracy(y_hat.cpu(), _y, topk=(1, 5))
        action_correct[_y] += (acc1 / 100.0) + (acc5 / 100.0)
        action_count[_y] += 1

        avg_acc1 += acc1
        avg_acc5 += acc5

        run_loss = avg_loss / (i + 1)
        run_acc1 = avg_acc1 / (i + 1)
        run_acc5 = avg_acc5 / (i + 1)
        progress_bar.set_postfix({
            'loss': '{:6.4f}'.format(run_loss),
            'acc1': '{:5.2f}%'.format(run_acc1),
            'acc5': '{:5.2f}%'.format(run_acc5),
        })

    accuracy_balance = torch.log1p(2 * action_count - action_correct)

    return (run_loss, run_acc1, run_acc5), accuracy_balance

def train(loader, model, optimizer, epoch):
    model.train()
    optimizer.zero_grad()

    avg_loss = 0.0
    n_samples = len(loader.dataset)
    progress_bar = tqdm(loader, disable=no_progress, position=0, leave=True)
    for i, (x, y) in enumerate(progress_bar):
        
        x = x.reshape(x.shape[0], x.shape[1], pose_dim).to(device)

        x = x.to(device)
        y = y.to(device)

        x = Variable(x, requires_grad=False)
        y = Variable(y, requires_grad=False)

        y_hat = model(x)

        loss = compute_loss(y_hat, y)
        loss.backward()

        avg_loss += loss.item()

        if (i + 1) % accumulate == 0 or (i + 1) == n_samples:
            if clip_norm:
                clip_grad_norm(classifier.parameters(), clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            avg_loss /= accumulate

            progress_bar.set_postfix({
                'loss': '{:6.4f}'.format(avg_loss),
            })
            
            if (i + 1) % log_every == 0:        
                print('Train Epoch {} [{}/{}]: Loss = {:6.4f}'.format(
                    epoch, i + 1, n_samples, avg_loss), file=log, flush=True)

            avg_loss = 0


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        base_dir = os.path.dirname(filename)
        best_filename = os.path.join(base_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def get_last_checkpoint(run_dir):
    last_checkpoint = os.path.join(run_dir, 'last_checkpoint.pth')
    if os.path.exists(last_checkpoint):
        return last_checkpoint

    def get_epoch(fname):
        epoch_regex = r'.*epoch_(\d+).pth'
        matches = re.match(epoch_regex, fname)
        return int(matches.groups()[0]) if matches else None

    checkpoints = glob.glob(os.path.join(run_dir, 'epoch_*.pth'))
    checkpoints = [(get_epoch(i), i) for i in checkpoints]
    last_checkpoint = max(checkpoints)[1]
    return last_checkpoint

# Resume training?

"""
run_dir = "results"
last_checkpoint = get_last_checkpoint(run_dir)
checkpoint = torch.load(last_checkpoint)
classifier.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
best_acc = checkpoint['best_accuracy']
start_epoch = checkpoint['epoch'] + 1
"""

best_acc = 0
start_epoch = 1
run_dir = "results"

progress_bar = trange(start_epoch, epochs + 1, initial=start_epoch, disable=no_progress)

for epoch in progress_bar:
    progress_bar.set_description('TRAIN [BestAcc1={:5.2f}]'.format(best_acc))
    train(train_loader, classifier, optimizer, epoch)
    
    progress_bar.set_description('EVAL')
    metrics, accuracy_balance = evaluate(test_loader, classifier)
    print('Test Epoch {}: Loss={:6.4f} Acc@1={:5.2f} Acc@5={:5.2f}'.format(epoch, *metrics),
          file=log, flush=True)
    
    current_acc1 = metrics[1]

    is_best = current_acc1 > best_acc
    best_acc = max(best_acc, current_acc1)
    
    checkpoint_filename = 'epoch_{:02d}.pth'.format(epoch)
    checkpoint_filename = os.path.join(run_dir, checkpoint_filename)
    
    save_checkpoint({
        'epoch': epoch,
        'best_accuracy': best_acc,
        'state_dict': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint_filename)
    