import torch as t
import numpy as np

def cross(x):
    p = np.array([[0,0,0,1,0]])
    x = x.numpy()
    x = np.exp(x)/np.exp(x).sum()
    r = p*np.log(x)
    r = r.sum()
    return r


x = t.tensor([[0.0,0.0,0.0,1.0,0.0]])
print(t.nn.functional.softmax(x))
f = t.nn.CrossEntropyLoss()
score = f(x,t.tensor([3]))
print(score)

print("----------")
print(cross(x))



