from torch.nn import Module,Flatten,Sequential,Linear,ReLU


class MyNeuralNetwork(Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.linear_relu_stack = Sequential(
            Linear(28*28,512),
            ReLU(),
            Linear(512,512),
            ReLU(),
            Linear(512,10)
        )

    def forward(self,x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
        


