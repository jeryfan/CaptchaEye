from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    transform = transforms.ToTensor()
    train_set = MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader