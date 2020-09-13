import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import EEG_Classifier 
from preprocess import label2np



torch.manual_seed(470)
torch.cuda.manual_seed(470)

device = 'cuda'
max_epoch = 40
learning_rate = 0.001
batch_size = 32

eeg_path = "EDF/190605C1_chunk.npy"
label_path = "EDF/190605C1_Hypno.txt"
data = np.load(eeg_path)
B, W, H, C = data.shape
data = np.reshape(data, (B, C, H, W)) 
label = label2np(label_path)
train_x = torch.Tensor(data[:4000])
train_y = torch.Tensor(label[:4000]).long()
test_x = torch.Tensor(data[4000:])
test_y = torch.Tensor(label[4000:]).long()
train_dataset = TensorDataset(train_x, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=32)
test_dataset = TensorDataset(test_x, test_y)
test_dataloader = DataLoader(test_dataset, batch_size=32)

training_process = True

model = EEG_Classifier()
model = model.to(device)
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
ckpt_dir = os.path.join(os.getcwd(), 'checkpoints')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

best_acc = 0.
ckpt_path = os.path.join(ckpt_dir, 'EEGmodel.pt')
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path)
    try:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_acc = ckpt['best_acc']
    except RuntimeError as e:
        print('wrong checkpoint')
    else:
        print("checkpoint is loaded ! \n current best accuracy : %.2f" % best_acc)

training_process = True
if training_process:
    it = 0
    train_losses = []
    test_losses = []
    for epoch in range(max_epoch):
        model.train()
        for inputs, labels in train_dataloader:
            it += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (logits.argmax(dim=1) == labels).float().mean()

            if it % 200 == 0:
                print('[epoch:{}, iteration:{}] train loacc : {:.4f} train accuracy: {:.4f}'.format(epoch+1, it, loss.item(), acc.item()))
        
        train_losses.append(loss)

        n = 0
        test_loss = 0.
        test_acc = 0.
        model.eval()
        for test_inputs, test_labels in test_dataloader:
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            logits = model(test_inputs)
            test_loss += F.cross_entropy(logits, test_labels, reduction='sum')
            test_acc += (logits.argmax(dim=1) == test_labels).float().sum().item()
            n += test_inputs.size(0)
        test_loss /= n
        test_acc /= n 
        test_losses.append(test_loss)
        print('[epoch:{}, iteration:{}] test_loss : {:.4f} test accuracy : {:.4f}'.format(epoch+1, it, test_loss, test_acc)) 

        # save checkpoint whenever there is improvement in performance
        if test_acc > best_acc:
            best_acc = test_acc
            # Note: optimizer also has states ! don't forget to save them as well.
            ckpt = {'model':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'best_acc':best_acc}
            torch.save(ckpt, ckpt_path)
            print('checkpoint is saved !')