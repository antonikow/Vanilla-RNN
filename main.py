import torchtext, torch
from torchtext.data import to_map_style_dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import os
import glob

torch.manual_seed(3)

batch_size = 200 
embed_size = 64
hidden_size =  128
max_words = 25 
lr = 0.001 
output_size = 4


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
path = "models"
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)
   print("The new directory is created!")

train_iter, test_iter = torchtext.datasets.AG_NEWS(root="data", split=('train', 'test'))
train_set = to_map_style_dataset(train_iter)
test_set = to_map_style_dataset(test_iter)
tokenizer = get_tokenizer("basic_english")
all = len(test_set) + len(train_set)

split = lambda x: (x[::2], x[1::2])
test_set, validation_set = split(test_set)
print("train samples", len(train_set)/all * 100, "%")
print("test samples", len(test_set)/all * 100, "%")
print("validation samp les", len(validation_set)/all * 100, "%")
print(len(test_set) + len(train_set) + len(validation_set))
print(len(test_set), len(train_set), len(validation_set))

    
def yield_tokens(data_iter):
    for data_set in data_iter:
        for _, text in data_set:
            yield tokenizer(text)
    

vocab = build_vocab_from_iterator(yield_tokens([train_set]), min_freq = 12, specials=["<unk>"], max_tokens=10000)
vocab.set_default_index(vocab["<unk>"])
vocab_size = len(vocab)
print(vocab_size, "vocab")
    

    

def collate_fn(batch):
    X, y = [], []
    for _label, _data in batch:       
        idxs = vocab(tokenizer(_data))[:max_words]
        if len(idxs) < max_words:
            idxs = idxs + [0] * (max_words - len(idxs))
        else:
            idxs = idxs[:max_words]
        X.append(idxs) 
        y.append(_label-1)
    X = torch.tensor(X, dtype=torch.int64)
    y = torch.tensor(y, dtype=torch.int64)
    return X, y
        
        

train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


X, _ = next(iter(validation_loader))

for x in X[:2]:
    for idx in x:
        token = idx.item()
        print(vocab.get_itos()[token], end=" ")
    print("\n")

embedding = torch.nn.Embedding(vocab_size, embed_size)

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.hidden_size = hidden_size
    combined = input_size + hidden_size
    self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.lin1 = nn.Linear(hidden_size, output_size)
    
  def init_hidden(self, batch_size):
    return torch.zeros((batch_size, self.hidden_size)) 

  def lin(self, x):
    x = self.lin1(x)
    return x

  def forward(self, x, h):
    combined = torch.cat((x, h), 1)
    next_h = self.tanh(self.i2h(combined)) 
    return next_h
 
model = RNN(embed_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr = lr) 
loss_fn = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience = 2, verbose = True)

n_epochs = 35
best_vloss = 1_000_000.

for epoch in range(n_epochs):
    # TRAINING
    model.train()
    tcorrect = 0.
    for i, data in enumerate(train_loader):
        X, y = data
        embedded = embedding(X)
        
        h = model.init_hidden(batch_size)
        for seq in range(max_words):
            h = model(embedded[:, seq,:], h)
        output = model.lin(h)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print (f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss:.6f}')
        
        # calculate train accuracy
        with torch.no_grad():
            _, predicted = torch.max(output, 1)
            tcorrect += (predicted == y).sum().item()
    with torch.no_grad():
        accuracy  =  tcorrect/len(train_set) * 100
        print("train accuracy:", accuracy,"%", end = " | ")
        
        # VALIDATION
        vcorrect = 0.
        vloss = 0.
        for i, data in enumerate(validation_loader):
            X, y = data
            embedded = embedding(X)
            
            h = model.init_hidden(batch_size)
            for seq in range(max_words):
                h = model(embedded[:, seq,:], h)
            output = model.lin(h)
            vloss += loss_fn(output, y).item()
            _, predicted = torch.max(output, 1)
            vcorrect += (predicted == y).sum().item()
            
        # calculate validation accuracy, loss and save the model
        accuracy = vcorrect/len(validation_set) * 100
        avg_vloss = vloss/len(validation_set)
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'models/model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)
            
        scheduler.step(avg_vloss)
        print("validation accuracy:", accuracy, "%", "avg_vloss:", avg_vloss)
        
        
# get path of best model
list_of_files = glob.glob('models/*')
latest_file = max(list_of_files, key=os.path.getctime)

# load the model
loaded_model = RNN(embed_size, hidden_size, output_size)
loaded_model.load_state_dict(torch.load(latest_file))
loaded_model.eval()
     
# TEST
test_correct = 0.
test_loss = 0.
with torch.no_grad():
    for i, data in enumerate(test_loader):
                X, y = data
                embedded = embedding(X)
                
                h = loaded_model.init_hidden(batch_size)
                for seq in range(max_words):
                    h = loaded_model(embedded[:, seq,:], h)
                output = loaded_model.lin(h)
                test_loss += loss_fn(output, y).item()
                _, predicted = torch.max(output, 1)
                test_correct += (predicted == y).sum().item()
                
    # calculate test accuracy and loss 
    accuracy = test_correct/len(test_set) * 100
    avg_testloss = test_loss/len(test_set)
    print("test accuracy:", accuracy, "%", "avg_testloss:", avg_testloss)