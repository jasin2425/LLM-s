import torch
import torch.nn.functional as F


def create_maps():
    # map  char -> index
    chars_map = {c: i + 1 for i, c in enumerate(chars)}
    chars_map['.'] = 0
    # map index -> char
    index_map = {i: c for c, i in chars_map.items()}
    return chars_map, index_map

g = torch.Generator().manual_seed(2147483647)

# list of names in a file
names = open('names.txt', 'r').read().splitlines()

# list of chars that exist in this dataset
chars = sorted(list(set("".join(names))))
chars_map,index_map=create_maps()

nof_bigrams=0
#input and labels
inputs,labels=[],[]
for name in names:
    name = ['.']+list(name)+['.']
    for ch1,ch2 in zip(name,name[1:]):
        nof_bigrams+=1
        indx1=chars_map[ch1]
        indx2=chars_map[ch2]
        inputs.append(indx1)
        labels.append(indx2)
inputs=torch.tensor(inputs)
labels=torch.tensor(labels)

#changing input shape to be able to do matrix multiplication
size=len(chars)+1
input_layer=F.one_hot(inputs,num_classes=size).float()
#weights array
W=torch.randn((size,size),generator=g,requires_grad=True)
loss=5.0
counter=0
#calculating negative log loss
for _ in range(500):
    counter+=1
    logits = input_layer @ W
    # softmax
    l_exp = logits.exp()
    # propabilites for each bigram
    output_layer = l_exp / l_exp.sum(1, keepdim=True)
    #loss with regularization
    loss=-output_layer[torch.arange(nof_bigrams),labels].log().mean() + 0.001*(W**2).mean()
    W.grad=None
    loss.backward()
    W.data-=W.grad*50
    if counter%100==0:
        print(loss)
print(loss)
g = torch.Generator().manual_seed(2147483647)

# drawing some words from model
nof_names=5
out = []
for _ in range(nof_names):
    indx = 0
    wrd = ""
    while True:
        x=F.one_hot(torch.tensor([indx]),27).float()
        p=x@W
        p=p.exp()
        p=p/p.sum(1, keepdim=True)
        indx = torch.multinomial(p, 1, replacement=True, generator=g).item()
        if indx == 0:
            out.append(wrd)
            break
        wrd += index_map[indx]

print(f'{nof_names} generated names:{out}')
