import numpy as np

with open('synset_words.txt', 'r') as f:
    lines = f.readlines()

all_info = []
for line in lines:
    info = []
    info.append(line[:9])
    info.append(line[9:-1])
    all_info.append(info)

all_info = np.array(all_info)
names = all_info[:, 0]
order = names.argsort()
all_info = all_info[order]

np.save('class_idx_name.npy', all_info)
