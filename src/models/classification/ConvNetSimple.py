import torch.nn as nn
import torch.nn.functional as F

from src.models.utils import num_flat_features


class ConvNetSimple(nn.Module):
    """ A simple convolution net to test classification performance on CIFAR10. """

    def __init__(self,
                 input_size=(32, 32),
                 number_of_input_channels=3,
                 number_of_classes=10):
        super(ConvNetSimple, self).__init__()
        # 1 or 3 - input image channel, 32 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(number_of_input_channels, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu4 = nn.ReLU()

        self.fc1 = nn.Linear(4096, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, number_of_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = F.max_pool2d(x, (2, 2), stride=[2, 2])

        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = F.max_pool2d(x, (2, 2), stride=[2, 2])

        x = x.view(-1, num_flat_features(x))

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
