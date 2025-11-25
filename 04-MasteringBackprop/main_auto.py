import random
from operator import index

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

g = torch.Generator().manual_seed(16092004)
NOF_LETTERS = 4 #number of letters from which we predict next
BATCH_SIZE = 128 #size of batch when we train
N_HIDDEN = 300 #number of neurons in hidden layers
N_EMBD = 10 #number of dimension for
MAX_STEPS=2000 #number of training cycles
def buildMaps(chars):
    indexMap = {index+1:letter for index,letter in enumerate(chars)}
    indexMap[0]='.'
    letterMap = {letter:index for index,letter in indexMap.items()}
    return indexMap,letterMap
def buildDataset(words,letterMap):
    X=[]
    Y=[]
    for word in words:
        word=word+'.'
        context=[0]*NOF_LETTERS
        for i in range(len(word)):
            Y.append(letterMap[word[i]])
            X.append(context)
            context=context[1:]+[letterMap[word[i]]]
    return torch.tensor(X),torch.tensor(Y)
def divideDataset(words,letterMap): #split data train 80 test 10 val 10
    random.shuffle(words)
    nofTrain=int(0.8*len(words))
    nofTest=int(0.9*len(words))
    xTrain,yTrain=buildDataset(words[:nofTrain],letterMap)
    xTest, yTest = buildDataset(words[nofTrain:nofTest],letterMap)
    xVal, yVal = buildDataset(words[nofTest:],letterMap)

    return xTrain,yTrain,xTest,yTest,xVal,yVal

class Linear:
    def __init__(self,fanIn,fanOut,bias=True):
        self.w=torch.randn((fanIn,fanOut),generator=g) * fanIn**-0.5 #kaiming init
        if bias:
            self.b=torch.zeros(fanOut)
        else:
            self.b=None
    def __call__(self,x):
        self.out=x@self.w
        if self.b is None:
            return self.out
        return self.out+self.b
    def parameters(self):
        return [self.w] + ([self.b] if self.b is None else [])
#like torch.nn.BatchNorm1D
class BatchNorm1d:
    def __init__(self,dim,eps=1e-05,momentum=0.1):
        self.eps=eps
        self.mom=momentum
        self.dim=dim
        self.gamma= torch.ones(dim)
        self.beta= torch.zeros(dim)
        self.training=True
        self.running_mean=torch.zeros(dim)
        self.running_var=torch.ones(dim)
    def __call__(self,x):
        if self.training:
            mean=x.mean(0,keepdim=True)
            var=x.var(0,keepdim=True)
        else:
            mean=self.running_mean
            var=self.running_var
        xhat = (x - mean) / (torch.sqrt(var + self.eps))  # normalize
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean=mean*self.mom+self.running_mean*(1-self.mom)
                self.running_var=var*self.mom+self.running_var*(1-self.mom)
        return self.out
    def parameters(self):
        return [self.gamma, self.beta]
class Tanh:
    def __call__(self, h):
        self.out=torch.tanh(h)
        return self.out
    def parameters(self):
        return []
def trainNetwork(layers,xtr,ytr,lr,C,parameters):
    for i in range(MAX_STEPS):

        #getting minibatch
        indexes = torch.randint(0,xtr.shape[0],(BATCH_SIZE,),generator=g)
        x=xtr[indexes]
        y=ytr[indexes]

        #creating input for a network
        emb=C[x]
        embcat=emb.view(-1,NOF_LETTERS*N_EMBD)
        output=embcat
        #forward pass
        for layer in layers:
            output=layer(output)
        loss=F.cross_entropy(output,y)

        #backward pass
        for p in parameters:
            p.grad=None

        loss.backward()
        #update parameters
        for p in parameters:
            p.data -= lr * p.grad

        #learning rate decoy
        if i>10000:
            lr=0.01
        if i % 100 == 0:
            print(f'{i}/{MAX_STEPS}: {loss.item():.4f}')
@torch.no_grad()
def split_loss(split,Xtr,Ytr,Xdev,Ydev,Xtest,Ytest,C,layers):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xtest, Ytest)
    }[split]
    emb=C[x]
    embcat=emb.view(-1,N_EMBD*NOF_LETTERS)
    out=embcat
    for layer in layers:
        out=layer(out)
    loss=F.cross_entropy(out,y)
    print(f'{split}, {loss.item()}')
@torch.no_grad()
def sampleFromModel(nofSamples,words,C,layers,indexMap):
    for _ in range (nofSamples):
        context=[0]*NOF_LETTERS
        name=""
        for word in words:
            emb=C[torch.tensor(context)]
            embcat=emb.view(-1,NOF_LETTERS*N_EMBD)
            out=embcat
            for layer in layers:
                out=layer(out)
            probs=F.softmax(out,dim=1)
            index=torch.multinomial(probs,1).item()
            name+=indexMap[index]
            context=context[1:] + [index]
            if index == 0:
                break
        print(name)


def main():
    with open('names.txt','r') as f:
        words=f.read().splitlines()
    chars=sorted(list(set(''.join( words))))
    vocab_size=len(chars)+1
    indexMap,letterMap=buildMaps(chars)
    xTrain,yTrain,xTest,yTest,xVal,yVal=divideDataset(words,letterMap)
    InitLR = 0.1

    # initzializing network

    C = torch.randn((vocab_size, N_EMBD), generator=g)
    # making 1 input layer 5 hidden layers(which consist of linear batch norm and tanh layer)
    layers=[
        Linear(N_EMBD*NOF_LETTERS,N_HIDDEN),BatchNorm1d(N_HIDDEN),Tanh(),
        Linear(N_HIDDEN,N_HIDDEN),BatchNorm1d(N_HIDDEN),Tanh(),
        Linear(N_HIDDEN, N_HIDDEN), BatchNorm1d(N_HIDDEN), Tanh(),
        Linear(N_HIDDEN, N_HIDDEN), BatchNorm1d(N_HIDDEN), Tanh(),
        Linear(N_HIDDEN, N_HIDDEN), BatchNorm1d(N_HIDDEN), Tanh(),
        Linear(N_HIDDEN, vocab_size), BatchNorm1d(vocab_size)
            ]
    with torch.no_grad():
        layers[-1].gamma*=0.1 #make logits less confident at initzialization more in 03-MLP UPGRADED
        for layer in layers[:-1]:
            if isinstance(layer,Linear):
                layer.w*=5/3 #kaiming for tanh
    parameters=[C] + [ parameter for layer in layers for parameter in layer.parameters()]
    for p in parameters:
        p.requires_grad=True

    #training data
    trainNetwork(layers,xTrain,yTrain,InitLR,C,parameters)
    for layer in layers:
        layer.training=False
    #split data
    # calculate loss
    split_loss('train',xTrain,yTrain,xVal,yVal,xTest,yTest,C,layers)
    split_loss('val',xTrain,yTrain,xVal,yVal,xTest,yTest,C,layers)
    split_loss('test',xTrain,yTrain,xVal,yVal,xTest,yTest,C,layers)

    sampleFromModel(20,words,C,layers,indexMap)


if __name__ == "__main__":
    main()