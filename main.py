# from projects.mnist import train_loader,test_loader,model,loss_func,optimizer,device
from projects.cifar10 import train_loader,test_loader,model,loss_func,optimizer,device
from libs import CaptchaEyeBase


print(device)
# captcha = CaptchaEyeBase(train_loader,model=model,loss_fn=loss_func,optimizer=optimizer,device=device)

# captcha.train(5)
# captcha.evaluate(test_loader)