import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sympy.codegen.ast import int32


def create_maps():
    # map  char -> index
    chars_map = {c: i + 1 for i, c in enumerate(chars)}
    chars_map['.'] = 0
    # map index -> char
    index_map = {i: c for c, i in chars_map.items()}
    return chars_map, index_map


def counts_array():
    # creating table of counts that store every biagram possible
    size=len(chars)+1
    count_tb = torch.zeros(size, size, dtype=torch.int64)
    for name in names:
        name_chars = ['.'] + list(name) + ['.']
        for ch1, ch2 in zip(name_chars, name_chars[1:]):
            ind_1 = chars_map[ch1]
            ind_2 = chars_map[ch2]
            count_tb[ind_1, ind_2] += 1
    return count_tb


def create_names(nof_names):
    # drawing some words from model
    out = []
    for _ in range(nof_names):
        indx = 0
        wrd = ""
        while True:
            p = P[indx, :]

            indx = torch.multinomial(p, 1, replacement=True, generator=g).item()
            if indx == 0:
                out.append(wrd)
                break
            wrd += index_map[indx]
    print(f'{nof_names} generated names:{out}')


def calculate_loss():
    # calculating loss
    log_loss = 0.0
    n = 0
    for w in names:
        w = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(w, w[1:]):
            loss = P[chars_map[ch1], chars_map[ch2]]
            log_loss += torch.log(loss)
            n += 1
    return log_loss, n


g = torch.Generator().manual_seed(2147483647)

# list of names in a file
names = open('names.txt', 'r').read().splitlines()

# list of chars that exist in this dataset
chars = sorted(list(set("".join(names))))

chars_map, index_map = create_maps()

count_tb = counts_array()

# creating table of propabilities
p = count_tb.float() + 1.0
P = p / p.sum(keepdim=True, dim=1)

create_names(8)

# negative log loss for clear loss evaluating
log_loss, n = calculate_loss()
nll = log_loss * -1
avg_log_loss = log_loss / n
print(f'negative log loss: {nll:.5f}, avg negative log loss: {avg_log_loss:.5f}')
