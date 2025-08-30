'''
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

g = torch.Generator().manual_seed(2147483647)

words = open('names.txt', 'r').read().splitlines()

#tensort that collects data about number of each letter (2 letters pair)
N=torch.zeros((27,27),dtype=torch.int64)

#list of chars
chars=sorted(list(set(''.join(words))))

#map of chars with their indexes
letter_map={s: i + 1 for i,s in enumerate(chars)}
letter_map['.']=0

#map of indexes with their letter
index_map={i:s for s,i in letter_map.items()}

#counting every pair in word
for w in words:
    chs=["."] + list(w)+ ["."]
    for ch1,ch2 in zip(chs,chs[1:]):
        N[letter_map[ch1],letter_map[ch2]]+=1

#P is 2 dimensional table(based on N) that has propability of each biagram (smoothed to avoid propability of 0)
P=N.float()+1.0
P/=P.sum(dim=1,keepdim=True)

#We draw letters based on propability in P
for i in range(10):
    indx = 0
    out = []
    while True:
        p=P[indx,:]
        indx=torch.multinomial(p,1,replacement=True,generator=g).item()
        out.append(index_map[indx])
        if indx==0:
            break
    print(''.join (out))

# calculating loss as negative log of cumulated propabilities (for each word)
log_likehood=0.0
n=0
for w in ["andrejq"]:
    chars=["."]+list(w)+["."]
    for ch1,ch2 in zip(chars,chars[1:]):
        nglog_prop=-torch.log(P[letter_map[ch1],letter_map[ch2]])
        log_likehood+=nglog_prop
        n+=1
        print(f'{ch1}{ch2} {P[letter_map[ch1],letter_map[ch2]]:.4f}  {nglog_prop:.4f}')
print(f'{log_likehood}')
print(f'({log_likehood/n}')

#input and output layers for neural network
#input layer
x_layer=[]
#output layer
y_layer=[]
for w in words[:1]:
    chars = ["."] + list(w) + ["."]
    for ch1,ch2 in zip(chars,chars[1:]):
        print (ch1,ch2)
        ch1_indx=letter_map[ch1]
        ch2_indx=letter_map[ch2]
        x_layer.append(ch1_indx)
        y_layer.append(ch2_indx)
print(x_layer,y_layer)

#we want 27 input neurons(for every letter)
x_layer=torch.tensor(x_layer)
x_layer=F.one_hot(x_layer,27).float()
print(x_layer)
'''



