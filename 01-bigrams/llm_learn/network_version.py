import torch
import torch.nn.functional as F


def create_maps():
    # map  char -> index
    chars_map = {c: i + 1 for i, c in enumerate(chars)}
    chars_map['.'] = 0
    # map index -> char
    index_map = {i: c for c, i in chars_map.items()}
    return chars_map, index_map
def make_input_output(data_type):
    nof_bigrams = 0
    # input and labels
    inputs, labels = [], []
    for i in data_type:
        name = ['.'] + list(names[i]) + ['.']
        for ch1, ch2 in zip(name, name[1:]):
            nof_bigrams += 1
            indx1 = chars_map[ch1]
            indx2 = chars_map[ch2]
            inputs.append(indx1)
            labels.append(indx2)
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    return nof_bigrams,inputs,labels
def calculate_loss(x,W,nof_bigrams,labels):
    logits=W[x]
    #softmax
    logits_exp=logits.exp()
    output_layer=logits_exp/logits_exp.sum(1,keepdim=True)
    #calculating loss with smoothing
    nll=-output_layer[torch.arange(nof_bigrams),labels].log().mean() + 0.001*(W**2).mean()
    return nll
def create_words():
    # drawing some words from model
    nof_names = 5
    out = []
    for _ in range(nof_names):
        indx = 0
        wrd = ""
        while True:
            x = F.one_hot(torch.tensor([indx]), 27).float()
            p = x @ W
            p = p.exp()
            p = p / p.sum(1, keepdim=True)
            indx = torch.multinomial(p, 1, replacement=True, generator=g).item()
            if indx == 0:
                out.append(wrd)
                break
            wrd += index_map[indx]
    print(f'{nof_names} generated names:{out}')
g = torch.Generator().manual_seed(2147483647)

# list of names in a file
names = open('names.txt', 'r').read().splitlines()
number_of_names = len(names)
permutations=torch.randperm(number_of_names,generator=g)
#divide data into test training and dev group
train=int(0.8*number_of_names)
test=int(0.1*number_of_names)
training_data=permutations[:train]
testing_data=permutations[train:train+test]
dev_data=permutations[train+test:]
print(training_data.shape,testing_data.shape)

# list of chars that exist in this dataset
chars = sorted(list(set("".join(names))))
chars_map, index_map = create_maps()

#creating input and label data
nof_bigrams,inputs,labels= make_input_output(training_data)

#training data

# changing input shape to be able to do matrix multiplication
size = len(chars) + 1

#creating weights array
W = torch.randn((size, size), generator=g, requires_grad=True)
counter = 0

# calculating negative log loss
for _ in range(500):
    counter += 1
    loss=calculate_loss(inputs,W,nof_bigrams, labels)
    W.grad = None
    loss.backward()
    W.data -= W.grad * 50
    if counter % 100 == 0:
        print(loss)
print(f'loss after training {loss}')

#testing weights
g = torch.Generator().manual_seed(2147483647)
nof_bigrams,test_input,test_labels=make_input_output(testing_data)
loss=calculate_loss(test_input,W,nof_bigrams, test_labels)
print(f'loss after testing {loss}')

create_words()

