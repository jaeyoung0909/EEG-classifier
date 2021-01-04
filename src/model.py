import torch.nn as nn
import torch.nn.functional as F

class EEG_Classifier(nn.Module):
      def __init__(self):
            super(EEG_Classifier, self).__init__()
            self.maxpool1 = nn.MaxPool2d((3,2), stride=(3,2))
            self.conv = nn.Conv2d(3, 50, kernel_size=(3, 3), stride=(1, 1))
            self.bn = nn.BatchNorm2d(50)
            self.maxpool2 = nn.MaxPool2d((2, 2), stride = (2, 2))
            self.linear1 = nn.Linear(50*39*3, 1000)
            self.linear2 = nn.Linear(1000, 1000)
            self.linear3 = nn.Linear(1000, 3)

      def forward(self, x):
            x = self.maxpool1(x)
            x = F.relu(self.bn(self.conv(x)))
            x = self.maxpool2(x)
            x = x.view(-1, 50*39*3)
            x = self.linear1(x)
            x = nn.Dropout(0.3)(x)
            x = self.linear2(x)
            x = nn.Dropout(0.3)(x)
            x = self.linear3(x)
            x = nn.Dropout(0.3)(x)
            x = nn.LogSoftmax()(x)

            return x