# pip install transformers
# pip  install openpyxl

import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
from transformers import BertTokenizer
from IPython.display import clear_output
import os
import pandas as pd
import re
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification
import gc
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from copy import deepcopy
from scipy.stats import entropy
import time
start = time.time()
PRETRAINED_MODEL_NAME = "bert-base-uncased"  
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
clear_output()
print("PyTorch 版本：", torch.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

df_train = pd.read_csv('../input/6666666666666666666/train.csv')

df_train = df_train.loc[:, ['text', 'target']]
df_train.columns = ['text_a', 'label']

seed = 696969
random.seed(seed)
random.shuffle(df_train.loc[:,'text_a'])
random.seed(seed)
random.shuffle(df_train.loc[:,'label'])
df_train=df_train.reset_index()

d={'label':df_train['label'].value_counts().index,'count':df_train['label'].value_counts()}
df_cat=pd.DataFrame(data=d).reset_index(drop=True)
print(df_cat)

df_train_train = df_train.loc[:, ['text_a', 'label']][:300]
# df_train_valid = df_train.loc[:, ['text_a', 'label']][300:6000]
# df_train_train=pd.concat([df_train_train,df_train_valid])
print("train樣本數：", len(df_train_train))
df_train_train.to_csv("train.tsv", sep="\t", index=False) 

d={'label':df_train_train['label'].value_counts().index,'count':df_train_train['label'].value_counts()}
df_cat=pd.DataFrame(data=d).reset_index(drop=True)
print(df_cat)

df_train_test = df_train.loc[:, ['text_a', 'label']][300:1300]
print("test樣本數：", len(df_train_test))
df_train_test.to_csv("testt.tsv", sep="\t", index=False)

d={'label':df_train_test['label'].value_counts().index,'count':df_train_test['label'].value_counts()}
df_cat=pd.DataFrame(data=d).reset_index(drop=True)
print(df_cat)

df_train_valid = df_train.loc[:, ['text_a', 'label']][1300:]
print("valid樣本數：", len(df_train_valid))
df_train_valid.to_csv("valid.tsv", sep="\t", index=False) 

d={'label':df_train_valid['label'].value_counts().index,'count':df_train_valid['label'].value_counts()}
df_cat=pd.DataFrame(data=d).reset_index(drop=True)
print(df_cat)


class FakeNewsDataset(Dataset):
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "testt", "valid"] 
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {0:0,1:1}
        self.tokenizer = tokenizer  

    def __getitem__(self, idx):# 定義回傳一筆訓練 / 測試數據的函式
        if self.mode == "submit":
            text_a, id= self.df.iloc[idx,:].values
            label_tensor = None
        else:
            text_a,label = self.df.iloc[idx, :].values
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)
         
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        segments_tensor = torch.tensor([0] * len_a ,dtype=torch.long)
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,batch_first=True)
    
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0,1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

def get_predictions(model, dataloader, compute_acc=False):
    model.eval()
    predictions = None
    correct = 0
    total = 0
    labelss = None
    probb= None
    with torch.no_grad():
        for data in dataloader:
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            Softmax = nn.Softmax(dim=1)
            probs = Softmax(logits.detach())
            _, predd = torch.max(probs.detach(), 1)
            
#             if compute_acc == False:
#                 return predd,probs,data[3]
            
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            if labelss is None:
                labelss = data[3]
            else:
                labelss = torch.cat((labelss, data[3]))
                
            if predictions is None:
                predictions = predd
            else:
                predictions = torch.cat((predictions, predd))
                
            if probb is None:
                probb = probs
            else:
                probb = torch.cat((probb, probs))
            
    if compute_acc:
        acc = correct / total
        return f1_score(predictions.data.cpu().numpy(), labelss.data.cpu().numpy()), acc
    
    return predictions,probb ,labelss

NUM_LABELS = 2

model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)


trainset = FakeNewsDataset("train", tokenizer=tokenizer)
trainloader = DataLoader(trainset, batch_size=32, collate_fn=create_mini_batch,shuffle=True)

testset = FakeNewsDataset("testt", tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=256, collate_fn=create_mini_batch)

model.train() # 訓練模式

optimizer = torch.optim.Adam(model.parameters(), lr=1e-6) 

EPOCHS = 50
record_max=0
record_max_f1_score=0
e=0

for epoch in range(EPOCHS):
    
    running_loss = 0.0

    for data in trainloader:
        model.train()
        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

        optimizer.zero_grad() 

        outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors,labels=labels)
        loss=outputs[0]
    
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 

    f1_train, acc = get_predictions(model, trainloader, compute_acc=True) 
    f1_test, tacc = get_predictions(model, testloader, compute_acc=True) 

    if(record_max_f1_score<f1_test):
        e=epoch
        record_max_f1_score=f1_test
        torch.save( deepcopy(model.state_dict()), "best.pt")
        
    if(record_max<tacc):
        record_max=tacc
        
    print("epoch:",epoch,"f1_test:",f1_test,"max:",record_max_f1_score)
#     print("epoch:",epoch,"f1_score:",f1_test,"f1_score_max:",record_max_f1_score,"\n") 
model.load_state_dict(torch.load("best.pt"))
model.eval()

validset = FakeNewsDataset("valid", tokenizer=tokenizer)
validloader = DataLoader(validset, batch_size=256, collate_fn=create_mini_batch)
pro, prediction, labell = get_predictions(model, validloader, compute_acc=False)

pro_wrong=[]
pro_cor=[]
pro_average=[]
pro_entropy=[]
pro_entropy_correct=[]
pro_entropy_wrong=[]

wrong_num=0      
for i in range(0,len(df_train_valid)):
    
    if(labell[i]!=pro[i]):  
        pro_wrong.append(100*torch.max(prediction[i].data))
        pro_entropy_wrong.append(entropy(prediction[i].data.cpu().detach().numpy()))
        
    if(labell[i]==pro[i]):
        pro_cor.append(100*torch.max(prediction[i].data))
        pro_entropy_correct.append(entropy(prediction[i].data.cpu().detach().numpy()))

    pro_average.append(100*torch.max(prediction[i].data))
    pro_entropy.append(entropy(prediction[i].data.cpu().detach().numpy()))

print(min(pro_cor))
print(min(pro_wrong))
print(min(pro_average))
print((sum(pro_cor)/len(pro_cor)))
print((sum(pro_wrong)/len(pro_wrong)))
print((sum(pro_average)/len(pro_average)))
print((sum(pro_entropy_correct)/len(pro_entropy_correct)))
print((max(pro_entropy_wrong)))
print((sum(pro_entropy_wrong)/len(pro_entropy_wrong)))
print((sum(pro_entropy)/len(pro_entropy)))
  
print("The time used to execute this is given below")
end = time.time()
print(end - start)    
    

bound=[0.50,0.50 ,0.4,0.3,0.3]
add=[]
num_add=0

for j in range(0,5):

  round_add=0  
  
  for i in range(0,len(df_train_valid)):

    if(entropy(prediction[i].cpu().detach().numpy())>bound[j]): 

      df_train.loc[7613] = df_train.loc[1300+i,['text_a', 'label']]
      df_train = df_train.drop(df_train.index[1300+i])
      df_train = df_train.reset_index(drop=True)
      num_add+=1
      round_add+=1

  print("num_add:",num_add)
  add.append(round_add)
  print(add)
  df=df_train_train
  df_add =df_train.loc[:,['text_a', 'label']][7613-round_add:7613]
  df_train_train=pd.concat([df,df_add])
  print("train樣本數：", len(df_train_train))
  df_train_train.to_csv("train.tsv", sep="\t", index=False) 

  df_train_valid = df_train.loc[:, ['text_a', 'label']][1300:7613-num_add]
  print("valid樣本數：", len(df_train_valid))
  df_train_valid.to_csv("valid.tsv", sep="\t", index=False) 


  trainset = FakeNewsDataset("train", tokenizer=tokenizer)
  trainloader = DataLoader(trainset, batch_size=64, collate_fn=create_mini_batch,shuffle=True)

  model.train() # 訓練模式

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-6) # 使用 Adam Optim 更新整個分類模型的參數

  EPOCHS = 30

  record_max=0
  record_max_f1_score=0
  
  for epoch in range(EPOCHS):
    
    running_loss = 0.0

    for data in trainloader:
        model.train()
        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

        optimizer.zero_grad() 

        outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors,labels=labels)
        loss=outputs[0]
    
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 

    f1_train, acc = get_predictions(model, trainloader, compute_acc=True) 
    f1_test, tacc = get_predictions(model, testloader, compute_acc=True) 

    if(record_max_f1_score<f1_test):
        e=epoch
        record_max_f1_score=f1_test
        torch.save( deepcopy(model.state_dict()), "best.pt")
        
    if(record_max<tacc):
        record_max=tacc
        
    print("epoch:",epoch,"f1_test:",f1_test,"max:",record_max_f1_score)
#     print("epoch:",epoch,"f1_score:",f1_test,"f1_score_max:",record_max_f1_score,"\n") 

  model.load_state_dict(torch.load("best.pt"))
  model.eval()
  
  validset = FakeNewsDataset("valid", tokenizer=tokenizer)
  validloader = DataLoader(validset, batch_size=256, collate_fn=create_mini_batch)
  pro, prediction, labell = get_predictions(model, validloader, compute_acc=False)

  pro_wrong=[]
  pro_cor=[]
  pro_average=[]
  wrong_num=0  
      
  for i in range(0,len(df_train_valid)):
    if(labell[i]!=pro[i]):  
      pro_wrong.append(100*torch.max(prediction[i].data))
      wrong_num+=1

    if(labell[i]==pro[i]):
      pro_cor.append(100*torch.max(prediction[i].data))

    pro_average.append(100*torch.max(prediction[i].data))

  print(min(pro_cor))
  print(min(pro_wrong))
  print(min(pro_average))
  print((sum(pro_cor)/len(pro_cor)))
  print((sum(pro_wrong)/len(pro_wrong)))
  print((sum(pro_average)/len(pro_average)))
    
print("The time used to execute this is given below")
end2 = time.time()
print(end2 - start)