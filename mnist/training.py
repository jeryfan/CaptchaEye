import os
import torch
from torch import nn, optim
from tqdm import tqdm
from model import MyModel,ConvModel
from utils import get_dataloaders

def train_model(epochs=5, lr=0.001, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, test_loader = get_dataloaders()
    best_acc = 0.0
    best_model_path = ""

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader,total=len(train_loader), desc=f"Epoch {epoch + 1}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        # 每轮后评估
        acc = evaluate(model, test_loader, device)
        print(f"✅ Test Accuracy after epoch {epoch + 1}: {acc * 100:.2f}%")

        # 保存当前轮模型
        model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)

        # 判断是否为最佳
        if acc > best_acc:
            # 删除旧最优模型
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)

            best_acc = acc
            best_model_path = model_path
            print(f"💾 Best model updated: {model_path}")
        else:
            # 删除非最优模型
            if os.path.exists(model_path):
                os.remove(model_path)

    print(f"\n✅ Training completed. Best model saved as: {best_model_path}")

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total