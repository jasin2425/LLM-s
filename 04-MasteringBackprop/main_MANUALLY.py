import random

import torch
import torch.nn.functional as F

NOF_LETTERS = 3  # from how many letters we want to predict next one
N_EMBD = 10  # dimensions of the C vector
N_HIDEEN = 300  # number of neurons in hidden layer
BATCH_SIZE = 128  # number of samples in one train
MAXSTEPS=2000
g = torch.Generator().manual_seed(2147483647)
def create_maps(chars):
    chars_map = {ch: indx + 1 for indx, ch in enumerate(chars)}
    chars_map['.'] = 0
    index_map = {indx: ch for ch, indx in chars_map.items()}
    return chars_map, index_map


def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')


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
    return Xtr, Ytr, Xtst, Ytst, Xdv, Ydv


def train_data(lr, Xtr, Ytr):
    with torch.no_grad():
        for k in range(MAXSTEPS):
            indexes = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,), generator=g)
            x=Xtr[indexes]
            y=Ytr[indexes]
            n=BATCH_SIZE
            # forward pass for manual backprop
            emb = C[x]
            embcat = emb.view(-1, N_EMBD * NOF_LETTERS)  # concatenate vectors
            hprebn = embcat @ W1 + b1
            # batch normalization
            bnmeani = 1/n*hprebn.sum(0,keepdim=True)
            bndiff=hprebn-bnmeani
            bndiff2=bndiff**2
            bnvar= 1 / (n-1) *   bndiff2.sum(0, keepdim=True) #formula for variance
            bnvar_inv=(bnvar+1e-5)**-0.5
            bnraw=bndiff*bnvar_inv
            hpreact = bngain * bnraw + bnbias  # we normalize andd the model is adjusting it for the best output
            h=torch.tanh(hpreact)
            logits = h @ W2 + b2
            # #calulate loss manual
            # #softmax
            # logit_maxes=logits.max(1,keepdim=True).values
            # norm_logits=logits-logit_maxes
            # counts = norm_logits.exp()
            # counts_sum=counts.sum(1,keepdim=True)
            # counts_sum_inv=counts_sum**-1
            # probs=counts*counts_sum_inv
            # #negative log loss
            # logprobs=probs.log()
            loss=F.cross_entropy(logits,y)

            for p in parameters:
                p.grad = None
            # for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
            #           norm_logits, logit_maxes, logits, h, hpreact, bnraw ,
            #           bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
            #           embcat, emb]:
            #     t.retain_grad()
            #loss.backward()

            # dlogprobs = torch.zeros_like(logprobs)
            # dlogprobs[range(n),y]=-1.0/len(y)
            #
            # dprobs = dlogprobs * 1 / probs
            # dcounts_sum_inv = (dprobs * counts).sum(1,
            #                                         keepdim=True)  # bc we broadcast counts_sum_inf we create 27 nodes so we should sum the gradients when backpropagating
            # dcounts_sum = dcounts_sum_inv * (-counts_sum ** -2)
            # dcounts = counts_sum_inv * dprobs + torch.ones_like(counts) * dcounts_sum
            # dnorm_logits = dcounts * norm_logits.exp()
            # dlogit_maxes = (-1.0 * dnorm_logits).sum(1, keepdim=True)
            #
            # max_indices = logits.max(1).indices
            # pom = F.one_hot(max_indices, num_classes=logits.shape[1])
            # dlogits = dnorm_logits + dlogit_maxes * pom
            #shorter version dloss/dlogits
            dlogits = F.softmax(logits)
            dlogits[range(n), y] -= 1
            dlogits *= 1 / n
            dh = dlogits @ W2.T
            dW2 = h.T @ dlogits
            db2 = dlogits.sum(0)

            dhpreact = (1.0 - h * h) * dh
            dbngain = (dhpreact * bnraw).sum(0, keepdim=True)
            #dbnraw = (dhpreact * bngain)
            dbnbias = (dhpreact).sum(0, keepdim=True)
            # dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
            # dbnvar = dbnvar_inv * -0.5 * (bnvar + 1e-5) ** -1.5
            # dbndiff2 = dbnvar * torch.ones_like(bndiff) * 1 / (n - 1)
            # dbndiff = dbndiff2 * 2 * bndiff + dbnraw * bnvar_inv
            # dbnmeani = (dbndiff * -1).sum(0, keepdim=True)
            #shorter version derivative dloss/dhprebn
            dhprebn = bngain * bnvar_inv / n * (
                        n * dhpreact - dhpreact.sum(0) - n / (n - 1) * bnraw * (dhpreact * bnraw).sum(0))
            dembcat = dhprebn @ W1.T
            dW1 = embcat.T @ dhprebn
            db1 = dhprebn.sum(0)
            demb = dembcat.view(emb.shape)
            dC = torch.zeros_like(C)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    cindex = x[i, j]
                    dC[cindex] += demb[i, j]
            # cmp('logprobs', dlogprobs, logprobs)
            # cmp('probs', dprobs, probs)
            # cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)
            # cmp('counts_sum', dcounts_sum, counts_sum)
            # cmp('counts', dcounts, counts)
            # cmp('norm_logits', dnorm_logits, norm_logits)
            # cmp('logit_maxes', dlogit_maxes, logit_maxes)
            # cmp('logits', dlogits, logits)
            # cmp('h', dh, h)
            # cmp('W2', dW2, W2)
            # cmp('b2', db2, b2)
            # cmp('hpreact', dhpreact, hpreact)
            # cmp('bngain', dbngain, bngain)
            # cmp('bnbias', dbnbias, bnbias)
            # cmp('bnraw', dbnraw, bnraw)
            #
            # cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
            # cmp('bnvar', dbnvar, bnvar)
            # cmp('bndiff2', dbndiff2, bndiff2)
            # cmp('bndiff', dbndiff, bndiff)
            # cmp('bnmeani', dbnmeani, bnmeani)
            # cmp('hprebn', dhprebn, hprebn)
            # cmp('embcat', dembcat, embcat)
            # cmp('W1', dW1, W1)
            # cmp('b1', db1, b1)
            # cmp('emb', demb, emb)
            # cmp('C', dC, C)
            # # backward pass
            # for p in parameters:
            #     p-=p.grad*lr
            lr = 0.1 if k < 100000 else 0.01
            grads=[dW1, dW2, db1, db2, dC, dbnbias, dbngain]
            for p,gradient in zip(parameters,grads):
                p-=gradient*lr
            if k % 100 == 0:
                print(f'{k}/{MAXSTEPS}: {loss.item():.4f}')


@torch.no_grad()
#calculating std and meand for batch normalization
def calc_mean_std():
    emb = C[Xtrening]
    embcat = emb.view(-1, NOF_LETTERS * N_EMBD)
    hpreact = embcat @ W1 + b1
    bnmean = hpreact.mean(0, keepdim=True)
    bnvar = hpreact.var(0, keepdim=True, unbiased=True)
    return bnmean, bnvar


@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtrening, Ytrening),
        'val': (Xdev, Ydev),
        'test': (Xtest, Ytest)
    }[split]
    emb = C[x]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5) ** -0.5 + bnbias
    h = torch.tanh(hpreact)
    # forward pass
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, y)
    print(f'{split}, {loss.item()}')


#split_loss('train')
#split_loss('val')
# reading data
words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
chars_map, index_map = create_maps(chars)
Xtrening, Ytrening, Xtest, Ytest, Xdev, Ydev = divide_dataset(words)

# INPUT LAYER
# initzialization of paremetrs:

C = torch.randn(len(chars) + 1, N_EMBD)  # embeding matrix
# layer 1:
W1 = torch.randn(N_EMBD * NOF_LETTERS, N_HIDEEN, generator=g) * (5 / 3) / (
        NOF_LETTERS * N_EMBD) ** 0.5  # normal random + kaiming init
b1 = torch.randn(N_HIDEEN, generator=g) * 0.1
# layer 2:
W2 = torch.randn(N_HIDEEN, len(chars) + 1, generator=g) * 0.1
b2 = torch.randn(len(chars) + 1, generator=g) * 0.1
# batchnorm layer doesn't zero for easier backprop debbuging:
bngain = torch.randn((1, N_HIDEEN))*0.1 + 1.0
bnbias = torch.randn((1, N_HIDEEN))*0.1

parameters = [W1, W2, b1, b2, C, bnbias, bngain]
for p in parameters:
    p.requires_grad = True

# Training network
train_data(0.1, Xtrening, Ytrening)
bnmean,bnvar=calc_mean_std()
split_loss('train')
split_loss('val')

#SAMPLING FROM MODEL
for _ in range(20):
    out = []
    context = [0] * NOF_LETTERS
    while True:
        emb = C[torch.tensor(context)]
        hpreact=emb.view(-1,N_EMBD*NOF_LETTERS)@W1+b1
        hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5) ** -0.5 + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2

        probs = F.softmax(logits, dim=1)
        indx = torch.multinomial(probs, 1, generator=g).item()
        context = context[1:] + [indx]
        out.append(indx)
        if indx == 0:
            break
    print(''.join(index_map[i] for i in out))