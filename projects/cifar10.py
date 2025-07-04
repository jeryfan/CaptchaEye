

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from libs import CaptchaEyeBase
from libs.model import Cifar10Model
from torch import nn,optim
import torch
import os


batch_size = 64
loss_func = nn.CrossEntropyLoss()
transform = transforms.ToTensor()
train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = Cifar10Model().to(device)
optimizer = optim.Adam(model.parameters())
