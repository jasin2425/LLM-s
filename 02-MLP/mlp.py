import random

import torch
import torch.nn.functional as F


def create_maps(chars):
    chars_map = {ch: indx + 1 for indx, ch in enumerate(chars)}
    chars_map['.'] = 0
    index_map = {indx: ch for ch, indx in chars_map.items()}
    return chars_map, index_map


# x and y are keeping indexes of chars
#we will have smth like this
# X=".ad"   Y="a"
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
def divide_dataset():
    # dividing dataset into train,test,dev
    indexes = torch.randperm(len(words))
    n1 = int(len(words) * 0.8)
    n2 = int(len(words) * 0.9)

    # from how many letters we want to predict next one
    nof_letters = 3
    # Inputs and labels
    random.shuffle(words)
    Xtr, Ytr = build_dataset(nof_letters, words[:n1])
    Xtst, Ytst = build_dataset(nof_letters, words[n1:n2])
    Xdv, Ydv = build_dataset(nof_letters, words[n2:])

def traing_data(lr):
    indexes = torch.randint(0,Xtr.shape[0],(128,),generator=g)
    emb = C[Xtr[indexes]]
    # HIDDEN LAYER
    h = torch.tanh(emb.view(-1, lv * nof_letters) @ W1 + b1)
    # forward pass
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

# reading data
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))

chars_map, index_map = create_maps(chars)


# shape_of_location_vecotor
lv = 10
g = torch.Generator().manual_seed(2147483647)  # for reproducibility



# INPUT LAYER
# embedding matrix (every letter has its 'location')
# For every letter group emb has their 'location' so...
# shape of this is [num_of_letters,num_of_previous_ltters,shape_of_location_vector)
C = torch.randn(len(chars) + 1, lv)
W1 = torch.randn(lv * nof_letters, 300, generator=g)
b1 = torch.randn(300, generator=g)
W2 = torch.randn(300, len(chars) + 1, generator=g)
b2 = torch.randn(len(chars) + 1, generator=g)
parameters = [W1, W2, b1, b2, C]
for p in parameters:
    p.requires_grad = True

'''finding best lr
lre=torch.linspace(-3,0,1000)
lrs=10**lre
lri=[]
'''
#Training network
losses=[]
step_counter=0
for i in range(20000):
    traing_data(0.15)
for i in range(15000):
    traing_data(0.1)
for i in range(15000):
    traing_data(0.05)
for i in range(15000):
    traing_data(0.01)

print(losses[-1])
emb=C[Xtst]
h = torch.tanh(emb.view(-1, lv * nof_letters) @ W1 + b1)
# forward pass
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytst)
print(f'loss for test data {loss}')

emb=C[Xdv]
h = torch.tanh(emb.view(-1, lv * nof_letters) @ W1 + b1)
# forward pass
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydv)
print(f'loss for dev data{loss}')


#SAMPLING FROM MODEL
for _ in range(20):
    out=[]
    context=[0]*nof_letters
    while True:
        emb=C[torch.tensor(context)]
        h = torch.tanh(emb.view(-1, lv * nof_letters) @ W1 + b1)
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
'''

