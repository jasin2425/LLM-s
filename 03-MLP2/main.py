import random

import torch
import torch.nn.functional as F

NOF_LETTERS=3 # from how many letters we want to predict next one
N_EMBD=10 #dimensions of the C vector
N_HIDEEN=300 #number of neurons in hidden layer
BATCH_SIZE=128 # number of samples in one train
def create_maps(chars):
    chars_map = {ch: indx + 1 for indx, ch in enumerate(chars)}
    chars_map['.'] = 0
    index_map = {indx: ch for ch, indx in chars_map.items()}
    return chars_map, index_map

# x and y are keeping indexes of chars
#we will have smth like this
# X=".ad", Y="a"
def build_dataset(nof_input_chars, names):
    X, Y = [], []
    for w in names :
        w = w + '.'
        single_x = [0] * nof_input_chars  #0 because it is the index o '.'
        for ch in w:
            new_index = chars_map[ch]
            Y.append(new_index)
            X.append(single_x)
            # updating input list
            single_x = single_x[1:] + [new_index]
    return torch.tensor(X), torch.tensor(Y)

# dividing dataset into train,test,dev
def divide_dataset(words):
    indexes = torch.randperm(len(words))
    n1 = int(len(words) * 0.8)
    n2 = int(len(words) * 0.9)
    # Inputs and labels
    random.shuffle(words)
    Xtr, Ytr = build_dataset(NOF_LETTERS, words[:n1])
    Xtst, Ytst = build_dataset(NOF_LETTERS, words[n1:n2])
    Xdv, Ydv = build_dataset(NOF_LETTERS, words[n2:])
    return Xtr,Ytr,Xtst,Ytst,Xdv,Ydv

def train_data(lr,Xtr,Ytr,losses):
    indexes = torch.randint(0,Xtr.shape[0],(BATCH_SIZE,),generator=g)
    # forward pass
    emb = C[Xtr[indexes]]
    embcat=emb.view(-1,N_EMBD*NOF_LETTERS) #concatenate vectors
    h = torch.tanh(embcat @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[indexes])
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    for p in parameters:
        p.data -=  lr* p.grad
    #lri.append(lrs[i])
    losses.append(loss.item())
@torch.no_grad()
def split_loss(split):
    x,y = {
        'train': (Xtrening,Ytrening),
        'val': (Xdev,Ydev),
        'test': (Xtest,Ytest)
    }[split]
    emb = C[x]
    embcat=emb.view(emb.shape[0],-1)
    h = torch.tanh(embcat @ W1 + b1)
    # forward pass
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(f'{split}, {loss.item()}')

# reading data
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))

chars_map, index_map = create_maps(chars)

Xtrening,Ytrening,Xtest,Ytest,Xdev,Ydev=divide_dataset(words)

# INPUT LAYER
# embedding matrix (every letter has its 'location')
# For every letter group emb has their 'location' so...
# shape of this is [NOF_LETTERS, num_of_previous_ltters, shape_of_location_vector)
g = torch.Generator().manual_seed(2147483647)
C = torch.randn(len(chars) + 1, N_EMBD) #embeding matrix
W1 = torch.randn(N_EMBD * NOF_LETTERS, N_HIDEEN, generator=g)*(5/3)/(NOF_LETTERS*N_EMBD)**0.5
b1 = torch.randn(N_HIDEEN, generator=g)*0.01
W2 = torch.randn(N_HIDEEN, len(chars) + 1, generator=g)*0.01
b2 = torch.randn(len(chars) + 1, generator=g)*0.01
parameters = [W1, W2, b1, b2, C]
for p in parameters:
    p.requires_grad = True

'''finding best learing rate
lre=torch.linspace(-3,0,1000)
lrs=10**lre
lri=[]
train, 2.049363374710083
val, 2.13407564163208
test, 2.139523983001709
'''
#Training network with decay
losses=[]
step_counter=0
for i in range(20000):
    train_data(0.15,Xtrening,Ytrening,losses)
for i in range(15000):
    train_data(0.1,Xtrening,Ytrening,losses)
for i in range(15000):
    train_data(0.05,Xtrening,Ytrening,losses)
for i in range(15000):
    train_data(0.01,Xtrening,Ytrening,losses)

split_loss('train')
split_loss('val')
split_loss('test')

#SAMPLING FROM MODEL
for _ in range(20):
    out=[]
    context=[0]*NOF_LETTERS
    while True:
        emb=C[torch.tensor(context)]
        h = torch.tanh(emb.view(-1, N_EMBD * NOF_LETTERS) @ W1 + b1)
        logits = h @ W2 + b2
        probs=F.softmax(logits,dim=1)
        indx=torch.multinomial(probs,1,generator=g).item()
        context=context[1:]+[indx]
        out.append(indx)
        if indx==0:
            break
    print(''.join(index_map[i] for i in (out)))
'''
not optimal way (more in readme)
counts=logits.exp()
prob=counts/counts.sum(1,keepdims=True)
print(prob.shape)
loss=-prob[torch.arange(67),Y].log().mean()
better cross entropy
'''

