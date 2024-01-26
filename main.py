from torchvision.datasets import MNIST,FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from Object import MyNeuralNetwork
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch import no_grad
import torch

train_data = MNIST(
    root='./res/',
    train=True,
    transform=ToTensor(),
    download=True
)

test_data = MNIST(
    root='./res/',
    train=False,
    transform=ToTensor(),
    download=True
)
train_data = DataLoader(train_data,batch_size=100)
test_data = DataLoader(test_data,batch_size=100)
mynn = MyNeuralNetwork()
lossFun = CrossEntropyLoss()
optimer = SGD(mynn.parameters(),lr = 0.001)
for t in range(10):
    print(f"-------------iteration:{t+1}-----------------")
    for batch,(X,Y) in enumerate(train_data):
        pred = mynn(X)
        loss_all = lossFun(pred,Y)
        loss_all.backward()
        optimer.step()
        optimer.zero_grad()
        if batch %100 == 0:
            print(f"已喂数据:{batch*100}   总损失:{loss_all.item()}")

    with no_grad():
        all_sample = len(test_data)*100
        correct    = 0 
        for batch,(X,Y) in enumerate(test_data):
            pred = mynn(X)
            correct += (pred.argmax(axis=1)==Y).type(torch.float).sum().item()

        print(f"正确率:  {(correct/all_sample)*100}%")
