import torch
from model import MyModel,ConvModel
from utils import get_dataloaders

def test_model(model_path='models/best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, test_loader = get_dataloaders()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            print((preds == labels).sum())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"🎯 Test Accuracy: {correct / total * 100:.2f}%")