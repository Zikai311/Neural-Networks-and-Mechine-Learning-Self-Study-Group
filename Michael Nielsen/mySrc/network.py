import numpy as np
import random

class Network:
    def __init__(self,sizes:list) -> None:
        self.num_layers=len(sizes)
        self.sizes=sizes
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x)for x,y in zip(sizes[:-1],sizes[1:])]
    
    def feedforword(self,a:np.ndarray):
        """Return the output of the network if "a" is input."""
        for w,b in zip(self.weights,self.biases):
            a=sigmoid(w@a+b)
        return a
        
        
        
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
        
if __name__=="__main__":
    net=Network([2,3,2])
    a=np.array([[0.9],[0.2]])
    print(a)
    b=net.feedforword(a)
    print(b)
    
    