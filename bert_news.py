# pip install transformers
# pip  install openpyxl
import torch
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

PRETRAINED_MODEL_NAME = "bert-base-chinese"  
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

print("PyTorch 版本：", torch.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

df_train = pd.read_excel('../input/555555/d.xlsx')

df_train = df_train.loc[:, ['title', 'label']]
df_train.columns = ['text_a', 'label']

seed = 50
random.seed(seed)
random.shuffle(df_train.loc[:,'text_a'])
random.seed(seed)
random.shuffle(df_train.loc[:,'label'])
df_train=df_train.reset_index()

df_train.drop(df_train.loc[df_train['label']=='社會'].index, inplace=True)
df_train.drop(df_train.loc[df_train['label']=='國際'].index, inplace=True)
df_train.drop(df_train.loc[df_train['label']=='生活'].index, inplace=True)
df_train.drop(df_train.loc[df_train['label']=='影劇'].index, inplace=True)
df_train=df_train.reset_index()

print(df_train[:100])


print("df_train:",len(df_train))
# print(df_train[:100])
# print(df_train[1500:1600])


print(df_train[:100])
print(df_train[1500:1600])

df_train_train = df_train.loc[:, ['text_a', 'label']][:100]
print("train樣本數：", len(df_train_train))
df_train_train.to_csv("train.tsv", sep="\t", index=False) 

df_train_test = df_train.loc[:, ['text_a', 'label']][100:1100]
print("test樣本數：", len(df_train_test))
df_train_test.to_csv("test.tsv", sep="\t", index=False)

df_train_valid = df_train.loc[:, ['text_a', 'label']][1100:3000]
print("valid樣本數：", len(df_train_valid))
df_train_valid.to_csv("valid.tsv", sep="\t", index=False) 


class FakeNewsDataset(Dataset):
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test","valid"] 
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {'政治':0,'體育':1,'財經':2, }
        self.tokenizer = tokenizer  

    def __getitem__(self, idx):# 定義回傳一筆訓練 / 測試數據的函式
        
        text_a,label = self.df.iloc[idx, :].values
        label_id = self.label_map[label]
        label_tensor = torch.tensor(label_id)
      
            
        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a ,dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len




# 這個函式的輸入 `samples` 是一個 list，裡頭的每個 element 都是剛剛定義的 `FakeNewsDataset` 回傳的一個樣本
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors,batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0,1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids


def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0

    with torch.no_grad():
      for data in dataloader:
        # print(data)
        if next(model.parameters()).is_cuda: 
            data = [t.to("cuda:0") for t in data if t is not None]

        tokens_tensors, segments_tensors, masks_tensors = data[:3]
        outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)
        logits = outputs[0]
        _, pred = torch.max(logits.data, 1)
        
        Softmax = nn.Softmax(dim=1)
        probs = Softmax(logits.detach())
        _, predd = torch.max(probs.detach(), 1)
        
        if compute_acc == False:
          # if(total<2):
          #   print("EEEEEEEEEEEEEEEEEEeeeee")
          #   print(pred,predd)
          #   print("%%%%%%%%%%%%%%%%%55")
          #   print(probs.detach())
          #   print("1111111111111111111111111111111111111")
          #   print(probs)
          #   print("\n")
          return predd,probs,data[3]
        if compute_acc:
              labels = data[3]
              total += labels.size(0)
              correct += (pred == labels).sum().item()
            
        if predictions is None:
            predictions = pred
        else:
            predictions = torch.cat((predictions, pred))
        
    if compute_acc:
        acc = correct / total
        return predictions, acc , correct , total 
    return predictions


PRETRAINED_MODEL_NAME = "bert-base-chinese"
NUM_LABELS = 3 

model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)

trainset = FakeNewsDataset("train", tokenizer=tokenizer)
trainloader = DataLoader(trainset, batch_size=64, collate_fn=create_mini_batch,shuffle=True)

testset = FakeNewsDataset("test", tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=256, collate_fn=create_mini_batch)

model.train() # 訓練模式

optimizer = torch.optim.Adam(model.parameters(), lr=1e-6) # 使用 Adam Optim 更新整個分類模型的參數

EPOCHS = 300
record_max=0
for epoch in range(EPOCHS):
    
  running_loss = 0.0

  for data in trainloader:
      
    tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

    optimizer.zero_grad() # 將參數梯度歸零
    
    # forward pass
    outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors, labels=labels)
    loss = outputs[0]

    # backward
    loss.backward()
    optimizer.step()

    
    running_loss += loss.item() # 紀錄當前 batch loss

    # torch.save(model, './model')

  _, acc,e,f = get_predictions(model, trainloader, compute_acc=True) # 計算分類準確率(train)
  _, tacc,a,b = get_predictions(model, testloader, compute_acc=True) # 計算分類準確率(test)

  print('[epoch %d] loss: %.3f, acc: %.3f, taac: %3f' %(epoch + 1, running_loss, acc, tacc))
  # print(a,b)

  if(record_max<tacc):
    record_max=tacc

  print(record_max)


validset = FakeNewsDataset("valid", tokenizer=tokenizer)
validloader = DataLoader(validset, batch_size=1900, collate_fn=create_mini_batch)
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

bound=[75,86,90,93,95,97,98,98.5]
add=[]
num_add=0

for j in range(0,8):

  round_add=0  
  
  for i in range(0,len(df_train_valid)):

    if(100*torch.max(prediction[i].data)<bound[j] and 100*torch.max(prediction[i].data)>30):

      df_train.loc[3000] = df_train.loc[1100+i,['text_a', 'label']]
      df_train = df_train.drop(df_train.index[1100+i])
      df_train = df_train.reset_index(drop=True)
      num_add+=1
      round_add+=1

  print("num_add:",num_add)
  add.append(round_add)
  print(add)
  df=df_train_train
  df_add =df_train.loc[:,['text_a', 'label']][3000-round_add:3000]
  df_train_train=pd.concat([df,df_add])
  print("train樣本數：", len(df_train_train))
  df_train_train.to_csv("train.tsv", sep="\t", index=False) 

  df_train_test = df_train.loc[:, ['text_a', 'label']][100:1100]
  print("test樣本數：", len(df_train_test))
  df_train_test.to_csv("test.tsv", sep="\t", index=False)

  df_train_valid = df_train.loc[:, ['text_a', 'label']][1100:3000-num_add]
  print("valid樣本數：", len(df_train_valid))
  df_train_valid.to_csv("valid.tsv", sep="\t", index=False) 


  trainset = FakeNewsDataset("train", tokenizer=tokenizer)
  trainloader = DataLoader(trainset, batch_size=64, collate_fn=create_mini_batch,shuffle=True)

  testset = FakeNewsDataset("test", tokenizer=tokenizer)
  testloader = DataLoader(testset, batch_size=256, collate_fn=create_mini_batch)

  model.train() # 訓練模式

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-6) # 使用 Adam Optim 更新整個分類模型的參數

  EPOCHS = 80

  record_max=0

  for epoch in range(EPOCHS):
      
    running_loss = 0.0

    for data in trainloader:
        
      tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

      optimizer.zero_grad() # 將參數梯度歸零
      
      # forward pass
      outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors, labels=labels)
      loss = outputs[0]

      # backward
      loss.backward()
      optimizer.step()

      
      running_loss += loss.item() # 紀錄當前 batch loss

      # torch.save(model, './model')

    _, acc,e,f = get_predictions(model, trainloader, compute_acc=True) # 計算分類準確率(train)
    _, tacc,a,b = get_predictions(model, testloader, compute_acc=True) # 計算分類準確率(test)

    print('[epoch %d] loss: %.3f, acc: %.3f, taac: %3f' %(epoch + 1, running_loss, acc, tacc))
    # print(a,b)

    if(record_max<tacc):
      record_max=tacc

    print(record_max)

  validset = FakeNewsDataset("valid", tokenizer=tokenizer)
  validloader = DataLoader(validset, batch_size=len(df_train_valid), collate_fn=create_mini_batch)
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


bound=[97.5,98,98.5]
add=[]
num_add=0

for j in range(0,3):

  round_add=0  
  
  for i in range(0,len(df_train_valid)):

    if(100*torch.max(prediction[i].data)<bound[j] and 100*torch.max(prediction[i].data)>30):

      df_train.loc[3000] = df_train.loc[1100+i,['text_a', 'label']]
      df_train = df_train.drop(df_train.index[1100+i])
      df_train = df_train.reset_index(drop=True)
      num_add+=1
      round_add+=1

  print("num_add:",num_add)
  add.append(round_add)
  print(add)
  df=df_train_train
  df_add =df_train.loc[:,['text_a', 'label']][3000-round_add:3000]
  df_train_train=pd.concat([df,df_add])
  print("train樣本數：", len(df_train_train))
  df_train_train.to_csv("train.tsv", sep="\t", index=False) 

  df_train_test = df_train.loc[:, ['text_a', 'label']][100:1100]
  print("test樣本數：", len(df_train_test))
  df_train_test.to_csv("test.tsv", sep="\t", index=False)

  df_train_valid = df_train.loc[:, ['text_a', 'label']][1100:3000-num_add]
  print("valid樣本數：", len(df_train_valid))
  df_train_valid.to_csv("valid.tsv", sep="\t", index=False) 


  trainset = FakeNewsDataset("train", tokenizer=tokenizer)
  trainloader = DataLoader(trainset, batch_size=64, collate_fn=create_mini_batch,shuffle=True)

  testset = FakeNewsDataset("test", tokenizer=tokenizer)
  testloader = DataLoader(testset, batch_size=256, collate_fn=create_mini_batch)

  model.train() # 訓練模式

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-6) # 使用 Adam Optim 更新整個分類模型的參數

  EPOCHS = 80

  record_max=0

  for epoch in range(EPOCHS):
      
    running_loss = 0.0

    for data in trainloader:
        
      tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

      optimizer.zero_grad() # 將參數梯度歸零
      
      # forward pass
      outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors, labels=labels)
      loss = outputs[0]

      # backward
      loss.backward()
      optimizer.step()

      
      running_loss += loss.item() # 紀錄當前 batch loss

      # torch.save(model, './model')

    _, acc,e,f = get_predictions(model, trainloader, compute_acc=True) # 計算分類準確率(train)
    _, tacc,a,b = get_predictions(model, testloader, compute_acc=True) # 計算分類準確率(test)

    print('[epoch %d] loss: %.3f, acc: %.3f, taac: %3f' %(epoch + 1, running_loss, acc, tacc))
    # print(a,b)

    if(record_max<tacc):
      record_max=tacc

    print(record_max)

  validset = FakeNewsDataset("valid", tokenizer=tokenizer)
  validloader = DataLoader(validset, batch_size=len(df_train_valid), collate_fn=create_mini_batch)
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