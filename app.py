import torch
from torch import nn



# parameters
batch_size=4
block_size=8


with open("A_battle_piece.txt",'r',encoding='utf-8') as f:
    text=f.read()
    print(len(text))
    # print(text[:100])
chars=sorted(set(text))
print(len(chars))
string_to_num={ch:i for i,ch in enumerate(chars)}
num_to_string={ch:i for ch,i in enumerate(chars)}
encode=lambda x: [string_to_num[i] for i in  x ]
decode=lambda x: ''.join([num_to_string[i] for i in  x ])


data=torch.tensor(encode(text),dtype=torch.long)
# print(data[:100])



#spliting of data
n=int(0.8*len(data))
train_split=data[:n]
test_spilt=data[n:]

def batch(x):
    curr_data=train_split if x=='train' else test_spilt
    ix=torch.tensor([c for i in curr_data[i]])


mode=input("either train or test")
get_batch=batch(mode)
