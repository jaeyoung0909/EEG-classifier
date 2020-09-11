import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

from tensorboardcolab import TensorBoardColab

torch.manual_seed(470)
torch.cuda.manual_seed(470)

# training & optimization hyper-parameters
max_epoch = 40
learning_rate = 0.001
batch_size = 32
device = 'cuda'

# model hyper-parameters
output_dim = 10 

# Boolean value to select training process
training_process = True

# initialize tensorboard for visualization
# Note : click the Tensorboard link to see the visualization of training/testing results
tbc = TensorBoardColab()

data_dir = os.path.join(gdrive_root, 'my_data')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class MyClassifier(nn.Module):
  def __init__(self):
      super(MyClassifier, self).__init__()
      self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
      self.bn1 = nn.BatchNorm2d(64)

      #layer1 
      self.conv1_1 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1,1), bias=False)
      self.bn1_1 = nn.BatchNorm2d(64)
      self.conv1_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1,1), bias=False)
      self.bn1_2 = nn.BatchNorm2d(64)

      self.conv1_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1,1), bias=False)
      self.bn1_3 = nn.BatchNorm2d(64)
      self.conv1_4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1,1), bias=False)
      self.bn1_4 = nn.BatchNorm2d(64)

      #layer2
      self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2,2), padding=(1,1), bias=False)
      self.bn2_1 = nn.BatchNorm2d(128)
      self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1,1), bias = False)
      self.bn2_2 = nn.BatchNorm2d(128)
      self.conv_bottle2_1 = nn.Conv2d(64, 128, kernel_size=(1,1), stride=(2,2), bias=False)
      self.bn_bottle2_1 = nn.BatchNorm2d(128)

      self.conv2_3 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1,1), bias=False)
      self.bn2_3 = nn.BatchNorm2d(128)
      self.conv2_4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1,1), bias=False)
      self.bn2_4 = nn.BatchNorm2d(128)

      #layer3
      self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2,2), padding=(1,1), bias=False)
      self.bn3_1 = nn.BatchNorm2d(256)
      self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1,1), bias = False)
      self.bn3_2 = nn.BatchNorm2d(256)
      self.conv_bottle3_1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(2,2), bias=False)
      self.bn_bottle3_1 = nn.BatchNorm2d(256)

      self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1,1), bias=False)
      self.bn3_3 = nn.BatchNorm2d(256)
      self.conv3_4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1,1), bias=False)
      self.bn3_4 = nn.BatchNorm2d(256)
      #layer4
      self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2,2), padding=(1,1), bias=False)
      self.bn4_1 = nn.BatchNorm2d(512)
      self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1), bias = False)
      self.bn4_2 = nn.BatchNorm2d(512)
      self.conv_bottle4_1 = nn.Conv2d(256, 512, kernel_size=(1,1), stride=(2,2), bias=False)
      self.bn_bottle4_1 = nn.BatchNorm2d(512)

      self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1), bias=False)
      self.bn4_3 = nn.BatchNorm2d(512)
      self.conv4_4 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1,1), bias=False)
      self.bn4_4 = nn.BatchNorm2d(512)
      self.dropout = nn.Dropout()


      self.linear = nn.Linear(512, 10)


  def forward(self, x):
      x = F.relu(self.bn1(self.conv1(x)))

      #layer1
      bottle = x
      x = F.relu(self.bn1_1(self.conv1_1(x)))
      x = self.bn1_2(self.conv1_2(x))
      x += bottle 
      x = F.relu(x)

      bottle = x
      x = F.relu(self.bn1_3(self.conv1_3(x)))
      x = self.bn1_4(self.conv1_4(x))
      x += bottle 
      x = F.relu(x)

      #layer 2
      bottle = x
      x = F.relu(self.bn2_1(self.conv2_1(x)))
      x = self.bn2_2(self.conv2_2(x))
      bottle = self.bn_bottle2_1 (self.conv_bottle2_1(bottle))
      x += bottle 
      x = F.relu(x)

      bottle = x
      x = F.relu (self.bn2_3(self.conv2_3(x)))
      x = self.bn2_4(self.conv2_3(x))
      x += bottle 
      x = F.relu(x)

      #layer 3
      bottle = x
      x = F.relu(self.bn3_1(self.conv3_1(x)))
      x = self.bn3_2(self.conv3_2(x))
      bottle = self.bn_bottle3_1 (self.conv_bottle3_1(bottle))
      x += bottle 
      x = F.relu(x)

      bottle = x
      x = F.relu (self.bn3_3(self.conv3_3(x)))
      x = self.bn3_4(self.conv3_3(x))
      x += bottle 
      x = F.relu(x)
      #layer 4
      bottle = x
      x = F.relu(self.bn4_1(self.conv4_1(x)))
      x = self.bn4_2(self.conv4_2(x))
      bottle = self.bn_bottle4_1 (self.conv_bottle4_1(bottle))
      x += bottle 
      x = F.relu(x)

      bottle = x
      x = F.relu (self.bn4_3(self.conv4_3(x)))
      x = self.bn4_4(self.conv4_3(x))
      x += bottle 
      x = F.relu(x)

      ##
      x = F.avg_pool2d(x, 4)
      # print(x.size())
      x = x.view(x.size(0), -1)
      # print(x.size())
      x = self.dropout(x)
      x = self.linear(x)
      return x

my_classifier = MyClassifier()
my_classifier = my_classifier.to(device)

# apply xavier initializer
def init_weights(m):
  if type(m) == nn.Linear or type(m) == nn.Conv2d:
    torch.nn.init.xavier_uniform(m.weight)

my_classifier.apply(init_weights)

# Print your neural network structure
print(my_classifier)

optimizer = optim.Adam(my_classifier.parameters(), lr=learning_rate)

ckpt_dir = os.path.join(gdrive_root, 'checkpoints')
if not os.path.exists(ckpt_dir):
  os.makedirs(ckpt_dir)
  
best_acc = 0.
ckpt_path = os.path.join(ckpt_dir, 'ResNet.pt')
if os.path.exists(ckpt_path):
  ckpt = torch.load(ckpt_path)
  try:
    my_classifier.load_state_dict(ckpt['my_classifier'])
    optimizer.load_state_dict(ckpt['optimizer'])
    best_acc = ckpt['best_acc']
  except RuntimeError as e:
      print('wrong checkpoint')
  else:    
    print('checkpoint is loaded !')
    print('current best accuracy : %.2f' % best_acc)

training_process = True
if training_process:
  it = 0
  train_losses = []
  test_losses = []
  for epoch in range(max_epoch):
    # train phase
    my_classifier.train()
    for inputs, labels in train_dataloader:
      it += 1

      # load data to the GPU.
      inputs = inputs.to(device)
      labels = labels.to(device)

      # feed data into the network and get outputs.
      logits = my_classifier(inputs)

      # calculate loss
      # Note: `F.cross_entropy` function receives logits, or pre-softmax outputs, rather than final probability scores.
      loss = F.cross_entropy(logits, labels)

      # Note: You should flush out gradients computed at the previous step before computing gradients at the current step. 
      #       Otherwise, gradients will accumulate.
      optimizer.zero_grad()

      # backprogate loss.
      loss.backward()

      # update the weights in the network.
      optimizer.step()

      # calculate accuracy.
      acc = (logits.argmax(dim=1) == labels).float().mean()

      if it % 2000 == 0:
        tbc.save_value('Loss', 'train_loss', it, loss.item())
        print('[epoch:{}, iteration:{}] train loss : {:.4f} train accuracy : {:.4f}'.format(epoch, it, loss.item(), acc.item()))

    # save losses in a list so that we can visualize them later.
    train_losses.append(loss)  

    # test phase
    n = 0.
    test_loss = 0.
    test_acc = 0.
    my_classifier.eval()
    for test_inputs, test_labels in test_dataloader:
      test_inputs = test_inputs.to(device)
      test_labels = test_labels.to(device)

      logits = my_classifier(test_inputs)
      test_loss += F.cross_entropy(logits, test_labels, reduction='sum').item()
      test_acc += (logits.argmax(dim=1) == test_labels).float().sum().item()
      n += test_inputs.size(0)

    test_loss /= n
    test_acc /= n
    test_losses.append(test_loss)
    tbc.save_value('Loss', 'test_loss', it, test_loss)
    print('[epoch:{}, iteration:{}] test_loss : {:.4f} test accuracy : {:.4f}'.format(epoch, it, test_loss, test_acc)) 

    tbc.flush_line('train_loss')
    tbc.flush_line('test_loss')

    # save checkpoint whenever there is improvement in performance
    if test_acc > best_acc:
      best_acc = test_acc
      # Note: optimizer also has states ! don't forget to save them as well.
      ckpt = {'my_classifier':my_classifier.state_dict(),
              'optimizer':optimizer.state_dict(),
              'best_acc':best_acc}
      torch.save(ckpt, ckpt_path)
      print('checkpoint is saved !')
    
tbc.close()