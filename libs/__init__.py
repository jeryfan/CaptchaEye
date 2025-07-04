from tqdm import tqdm
import os,torch


class CaptchaEyeBase:

    def __init__(self,data_loader=None,model=None,optimizer=None,loss_fn=None,device=None):
        self.data_loader = data_loader
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn


    def train(self,epochs,save_path="outputs/model.pth"):
        
        for epoch in range(epochs):
            loop = tqdm(self.data_loader,total=len(self.data_loader),desc=f"Epoch: {epoch+1}/{epochs}")
            self.model.train()
            for image,label in loop:
                image,label = image.to(self.device),label.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.loss_fn(output,label)
                loss.backward()
                self.optimizer.step()
                loop.set_postfix(loss=loss.item(),refresh=True)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        torch.save(self.model.state_dict(), save_path)

    def evaluate(self,test_loader):
        self.model.eval()
        correct = 0
        total = 0
        test_loader = tqdm(test_loader,total=len(test_loader),desc="Evaluate")
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                test_loader.set_description(f"Acc: {correct / total:.2f}")

        return correct / total
            







