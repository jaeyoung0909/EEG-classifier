import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from model import EEG_Classifier 
from preprocess import label2np


class EEG_Model:
    def __init__(self):
        self._train_dataloader = None
        self._test_dataloader = None

    def fit(self, eeg_path, label_path, batch_size = 32):
        data = np.load(eeg_path)[:10796]
        B, W, H, C = data.shape
        data = np.reshape(data, (B, C, H, W)) 
        label = label2np(label_path)
        input = torch.Tensor(data)
        label = torch.Tensor(label).long()
        dataset = TensorDataset(input, label)

        length = len(dataset)
        train_size = int(length*0.8)
        test_size = length - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        self._train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        self._test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    def train(self, max_epoch = 100, learning_rate = 0.001, ckpt = 'EEGmodel.pt', device = 'cuda'):
        torch.manual_seed(470)
        torch.cuda.manual_seed(470)
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
        ckpt_path = os.path.join(ckpt_dir, ckpt)
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

        it = 0
        for epoch in range(max_epoch):
            model.train()
            for inputs, labels in self._train_dataloader:
                it += 1
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = (logits.argmax(dim=1) == labels).float().mean()

            n = 0
            test_loss = 0.
            test_acc = 0.
            model.eval()
            for test_inputs, test_labels in self._test_dataloader:
                test_inputs = test_inputs.to(device)
                test_labels = test_labels.to(device)
                logits = model(test_inputs)
                test_loss += F.cross_entropy(logits, test_labels, reduction='sum')
                test_acc += (logits.argmax(dim=1) == test_labels).float().sum().item()
                n += test_inputs.size(0)
            test_loss /= n
            test_acc /= n 
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

    def predict(dataloader, ckpt_file_name, device = 'cuda'):
        model = EEG_Classifier()
        model = model.to(device)
        ckpt_dir = os.path.join(os.getcwd(), 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        ckpt_path = os.path.join(ckpt_dir, ckpt_file_name)
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
        else:
            RuntimeError("ERROR: no such model file name. Please check .pt file for trained model weight")
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss += F.cross_entropy(logits, labels, reduction='sum')
            acc += (logits.argmax(dim=1) == test_labels).float().sum.item()
            n += test_inputs.size(0)
        loss /= n
        acc /= n 
        print('loss : {:.4f} accuracy : {:.4f}'.format(loss, acc))
