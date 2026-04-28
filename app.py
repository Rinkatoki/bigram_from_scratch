import torch
from torch import nn
from torch.nn import functional as F

# parameters
batch_size=4
block_size=8
eval_iters=250
learning_rate = 6e-4
max_iters=1000


with open("A_battle_piece.txt",'r',encoding='utf-8') as f:
    text=f.read()
    print(len(text))
    # print(text[:100])
chars=sorted(set(text))
print(len(chars))
vocab_size=len(chars)
string_to_num={ch:i for i,ch in enumerate(chars)}
num_to_string={ch:i for ch,i in enumerate(chars)}
encode=lambda x: [string_to_num[i] for i in  x ]
decode=lambda x: ''.join([num_to_string[i] for i in  x ])


data=torch.tensor(encode(text),dtype=torch.long)
# print(data[:100])



#spliting of data
n=int(0.8*len(data))
train_split=data[:n]
test_splitt=data[n:]

def batch(z):
    curr_data=train_split if z=='train' else test_splitt
    ix=torch.randint(len(curr_data)-block_size,(batch_size,))
    x=torch.stack([curr_data[i:i+block_size] for i in ix])
    y=torch.stack([curr_data[i+1:i+block_size+1] for i in ix])
    return x,y


x,y=batch("train")
# print("input:",x)
# print("target",y)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class BigramModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding_table=nn.Embedding(vocab_size,vocab_size)
    def forward(self,index,target=None):
        logits=self.embedding_table(index)
        if target==None:
            loss =None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits,loss
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index
    
model=BigramModel(vocab_size)
context = torch.zeros((1,1), dtype=torch.long)
generated_chars = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['test']:.3f}")

    # sample a batch of data
    xb, yb =batch('train')

    # evaluate the loss
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # print(model.embedding_table.weight.grad)
    optimizer.step()
print(loss.item())

