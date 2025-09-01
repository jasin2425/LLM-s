import random

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
size=len(chars)+1

nof_trigrams=0
#input and labels
inputs,labels=[],[]
for name in names:
    name = ['.']+list(name)+['.']
    for ch1,ch2,ch3 in zip(name,name[1:],name[2:]):
        nof_trigrams+=1
        indx1=chars_map[ch1]
        indx2=chars_map[ch2]
        input_indx=indx1*size+indx2
        indx3=chars_map[ch3]
        inputs.append(input_indx)
        labels.append(indx3)
inputs=torch.tensor(inputs)
labels=torch.tensor(labels)
print(f'{inputs},  {labels}')

#changing input shape to be able to do matrix multiplication
input_layer=F.one_hot(inputs,num_classes=size*size).float()
#weights array
W=torch.randn((size*size,size),generator=g,requires_grad=True)
counter=0
for _ in range(10):
    counter+=1
    #calculating negative log loss
    logits=input_layer@W
    #softmax
    logits_exp=logits.exp()
    output_layer=logits_exp/logits_exp.sum(1,keepdim=True)
    loss = -output_layer[torch.arange(nof_trigrams), labels].log().mean()
    W.grad=None
    loss.backward()
    W.data-=W.grad*3
    print(loss)
print(loss)
g = torch.Generator().manual_seed(2147483647)

# drawing some words from model
nof_names=5
out = []
for _ in range(nof_names):
    indx = 0
    indx2 = random.randint(1, 27)
    wrd = ""
    while True:
        x=F.one_hot(torch.tensor([indx*27+indx2]),size*size).float()
        p=x@W
        p=p.exp()
        p=p/p.sum(1, keepdim=True)
        indx3 = torch.multinomial(p, 1, replacement=True, generator=g).item()
        wrd += index_map[indx3]
        indx=indx2
        indx2=indx3
        if indx3 == 0:
            out.append(wrd)
            break

print(f'{nof_names} generated names:{out}')
