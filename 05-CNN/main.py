import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# consts
NOF_LETTERS = 8  # from how many letters we want to predict next one
N_EMBD = 24  # dimensions of the C vector
N_HIDEEN = 300  # number of neurons in hidden layer
BATCH_SIZE = 128  # number of samples in one train
g = torch.Generator().manual_seed(9876543987)


# x and y are keeping indexes of chars
# we will have smth like this
# X=".ad", Y="a"
def build_dataset(nof_input_chars, names):
    X, Y = [], []
    for w in names:
        w = w + '.'
        single_x = [0] * nof_input_chars  # 0 because it is the index o '.'
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
        self.weight = torch.randn((fan_in, fan_out), generator=g) * fan_in ** (-0.5)  # kaiming init
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


# like torch.nn.BatchNorm1D
class BatchNorm1d:
    def __init__(self, dim, eps=1e-05, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # batch shifting
        self.beta = torch.zeros(dim)
        self.gamma = torch.ones(dim)
        self.runningMean = torch.zeros(dim)
        self.runningVar = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim==2:
                dim=0
            elif x.ndim==3: #3 dimension of x
                dim=(0,1)
            xmean = x.mean(dim, keepdim=True)  # batch mean (in wn we want to it for whole batch not every pair)
            xvar = x.var(dim, keepdim=True)  # batch variance
        else:
            xmean = self.runningMean
            xvar = self.runningVar
        xhat = (x - xmean) / (torch.sqrt(xvar + self.eps))  # normalize
        self.out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.runningMean=(1-self.momentum)*self.runningMean + self.momentum*xmean
                self.runningVar = (1 - self.momentum) * self.runningVar + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# like torch.Tanh
class Tanh:
    def __call__(self, h):
        self.out = torch.tanh(h)
        return self.out

    def parameters(self):
        return []
#like nn.embeding
class Embedding:
    def __init__(self,num_embeddings,embedding_dim):
        self.weights=torch.randn((num_embeddings,embedding_dim),generator=g)
    def __call__(self,X_batch):
        self.out= self.weights[X_batch]
        return self.out
    def parameters(self):
        return [self.weights]
class FlattenWaveNet:
    def __init__(self,n):
        self.n=n #number of letters that we want to merge ( WAVENET )
    def __call__(self, X):
        #batch size, nofletters,dimensions
        B,L,D=X.shape
        X=X.view(B,L//self.n,D*self.n)
        #for the last pair (last layer)
        if X.shape[1]==1:
            X=X.squeeze(1)
        self.out=X
        return self.out
    def parameters(self):
        return []
class Sequential:
    def __init__(self,layers):
        self.layers=layers
    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        self.out=x
        return self.out
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

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
    maxSteps = 20000
    lossi = []
    for i in range(maxSteps):
#GETTING MINIBATCH
        indexes = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
        X_batch = Xtr[indexes]
        Y_batch = Ytr[indexes]

        x=X_batch
#FORWARD PASS
        logits=model(x)
        loss = F.cross_entropy(logits, Y_batch)

#BACKWARD PASS
        for p in parameters:
            p.grad = None
        loss.backward()

#LEARNING RATE DECOY
        if i > 15000:
            lr = 0.01
        for p in parameters:
            p.data -= lr * p.grad

        lossi.append(torch.log10(loss))
    plt.plot(torch.tensor(lossi).view(-1,200).mean(1))
    plt.show()

@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtrening, Ytrening),
        'val': (Xdev, Ydev),
        'test': (Xtest, Ytest)
    }[split]
#FORWARD PASS
    logits=model(x)
#CALULATING LOSS
    loss = F.cross_entropy(logits, y)
    print(f'{split}, {loss.item()}')


@torch.no_grad()
def sampleFromModel(nofSamples):
    for _ in range(nofSamples):
        starting = [0] * NOF_LETTERS
        name = ""
        while True:
            x=torch.tensor([starting])
        #FORWARD PASS
            logits = model(x)
        #CALCULATE PROBABILITIES
            probs = F.softmax(logits, dim=1)
            newindex = torch.multinomial(probs, 1).item()
            name += index_map[newindex]
            starting = starting[1:] + [newindex]
            if newindex == 0:
                break
        print(name)



# reading data
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
vocab_size = len(chars) + 1
chars_map, index_map = create_maps(chars)
Xtrening, Ytrening, Xtest, Ytest, Xdev, Ydev = divide_dataset(words)
InitLR = 0.1

# INITZIALIZING NETWORK
model =Sequential( [
    Embedding(vocab_size,N_EMBD),
    FlattenWaveNet(2),
    Linear(N_EMBD * 2, N_HIDEEN), BatchNorm1d(N_HIDEEN), Tanh(),
    FlattenWaveNet(2),
    Linear(N_HIDEEN*2, N_HIDEEN), BatchNorm1d(N_HIDEEN), Tanh(),
    FlattenWaveNet(2),
    Linear(N_HIDEEN*2, vocab_size)
])

# initzialize data
with torch.no_grad():
    model.layers[-1].weight *= 0.1  # make logits smaller and model less confident
    for i in model.layers[:-1]:
        if isinstance(i, Linear):
            i.weight *= 5 / 3  # kaiming init it is important cause tanh will squish our layer outputs so we need smth to fight it
parameters = model.parameters()
for p in parameters:
    p.requires_grad = True

# training network
trainNetwork(Xtrening, Ytrening, InitLR)

for layer in model.layers:
    layer.training = False

# calculate loss
split_loss('train')
split_loss('val')
split_loss('test')

sampleFromModel(20)
