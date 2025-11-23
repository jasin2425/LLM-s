import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
#consts
NOF_LETTERS = 3  # from how many letters we want to predict next one
N_EMBD = 10  #dimensions of the C vector
N_HIDEEN = 300  #number of neurons in hidden layer
BATCH_SIZE = 128  # number of samples in one train


# x and y are keeping indexes of chars
#we will have smth like this
# X=".ad", Y="a"
def build_dataset(nof_input_chars, names):
    X, Y = [], []
    for w in names:
        w = w + '.'
        single_x = [0] * nof_input_chars  #0 because it is the index o '.'
        for ch in w:
            new_index = chars_map[ch]
            Y.append(new_index)
            X.append(single_x)
            # updating input list
            single_x = single_x[1:] + [new_index]
    return torch.tensor(X), torch.tensor(Y)


def create_maps(chars):
    chars_map = {ch: indx + 1 for indx, ch in enumerate(chars)}
    chars_map['.'] = 0
    index_map = {indx: ch for ch, indx in chars_map.items()}
    return chars_map, index_map


# like torch.nn.linear
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) * fan_in ** (-0.5)
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out=x@self.weight
        if self.bias is not None:
             self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


#like torch.nn.BatchNorm1D
class BatchNorm1d:
    def __init__(self, dim, eps=1e-05, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        #batch shifting
        self.beta = torch.zeros(dim)
        self.gamma = torch.ones(dim)
        self.runningMean = torch.zeros(dim)
        self.runningVar = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True) #batch mean
            xvar = x.var(0, keepdim=True) #batch variance
        else:
            xmean = self.runningMean
            xvar = self.runningVar
        xhat = (x - xmean) / (torch.sqrt(xvar+ self.eps)) #normalize
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.runningMean = (1 - self.momentum) * self.runningMean + self.momentum * xmean
                self.runningVar = (1 - self.momentum) * self.runningVar + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


#like torch.Tanh
class Tanh:
    def __call__(self, h):
        self.out = torch.tanh(h)
        return self.out

    def parameters(self):
        return []


# dividing dataset into train,test,dev
def divide_dataset(words):
    torch.randperm(len(words))
    n1 = int(len(words) * 0.8)
    n2 = int(len(words) * 0.9)
    # Inputs and labels
    random.shuffle(words)
    Xtr, Ytr = build_dataset(NOF_LETTERS, words[:n1])
    Xtst, Ytst = build_dataset(NOF_LETTERS, words[n1:n2])
    Xdv, Ydv = build_dataset(NOF_LETTERS, words[n2:])
    return Xtr, Ytr, Xtst, Ytst, Xdv, Ydv


def trainNetwork(Xtr, Ytr, lr):
    maxSteps = 2000
    lossi = []
    ud = []
    for i in range(maxSteps):
        #getting minibatch
        indexes = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
        X_batch = Xtr[indexes]
        Y_batch = Ytr[indexes]

        #create array of embeedings
        emb = C[X_batch]
        emcat = emb.view(-1, NOF_LETTERS * N_EMBD)
        #forward pass
        for layer in layers:
            emcat = layer(emcat)
        loss = F.cross_entropy(emcat, Y_batch)
        #backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        if i > 10000:
            lr = 0.01
        for p in parameters:
            p.data -= lr * p.grad
@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtrening, Ytrening),
        'val': (Xdev, Ydev),
        'test': (Xtest, Ytest)
    }[split]
    emb = C[x]
    x = emb.view(-1,NOF_LETTERS*N_EMBD)
    for layer in layers:
        x=layer(x)

    loss = F.cross_entropy(x, y)

    print(f'{split}, {loss.item()}')

@torch.no_grad()
def sampleFromModel(nofSamples):
    for _ in range(nofSamples):
        starting=[0]*NOF_LETTERS
        name=""
        while True:
            emb = C[torch.tensor([starting])]
            x=emb.view(-1,NOF_LETTERS*N_EMBD)
            for layer in layers:
                x=layer(x)
            logits=x
            probs=F.softmax(logits,dim=1)
            newindex=torch.multinomial(probs,1).item()
            name+=index_map[newindex]
            starting=starting[1:] + [newindex]
            if newindex == 0:
                break
        print(name)




n_embd = 10  #dimensions of character-embedding vectors
n_hidden = 100  #number of neurons in a hidden layer
g = torch.Generator().manual_seed(9876543987)

# reading data
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
vocab_size = len(chars)+1
chars_map, index_map = create_maps(chars)
Xtrening, Ytrening, Xtest, Ytest, Xdev, Ydev = divide_dataset(words)
InitLR=0.1

#initzializing network
C = torch.randn((vocab_size, n_embd), generator=g)
#making 1 input layer 5 hidden layers(which consist of linear and non linear) and 1 output layer
layers = [
    Linear(n_embd * NOF_LETTERS, n_hidden),BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden),BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden),BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden),BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, n_hidden),BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),BatchNorm1d(vocab_size)
]

#initzialize data
with torch.no_grad():
    layers[-1].gamma *= 0.1  #make logits smaller and model less confident
    for i in layers[:-1]:
        if isinstance(i, Linear):
            i.weight *= 5 / 3  #kaiming init it is important cause tanh will squish our layer outputs so we need smth to fight it
parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
    p.requires_grad = True

#training network
trainNetwork(Xtrening,Ytrening,InitLR)

for layer in layers:
  layer.training = False
#calculate loss
split_loss('train')
split_loss('val')
split_loss('test')

sampleFromModel(20)


