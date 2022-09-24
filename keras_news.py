import os
os.environ['PYTHONHASHSEED'] = '1'

import numpy as np
import tensorflow 
import random

random.seed(1)
np.random.seed(1)
tensorflow .random.set_seed(1)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import pandas as pd
from tensorflow import keras
import jieba.posseg as pseg
import jieba.analyse

from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding,LSTM, concatenate, Dense,Dropout,SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from scipy.stats import entropy

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_colwidth', -1) 

outlist=[]
words=[500,10000,12500,15000,20000,25000,35000,50000,70000]
max_sequence_len=[10,20,25,30,40,50]
vector=[32,64,128,192,256,512]
epochlist=[12,15,17,18,20,22,23,25]
batchlist=[16,32,48,64,96,128,256,512]
droplist=[0.1,0.15,0.2,0.25,0.3,0.5]

for j in range(0,1):

  MAX_NUM_WORDS = 360000
  MAX_SEQUENCE_LENGTH = 70
  NUM_CLASSES = 3
  NUM_EMBEDDING_DIM = 128
  NUM_LSTM_UNITS =128
  use_col='title'
   
  train = pd.read_excel('./d.xlsx')
  out=pd.read_excel('./d.xlsx')
  cols = [use_col, 'label']
  train = train.loc[:,cols]
  out=out.loc[:,cols]

  train=train.dropna()
  train=train.reset_index()

  random.seed(12345)
  random.shuffle(train.loc[:6800,use_col])
  random.seed(12345)
  random.shuffle(train.loc[:6800,'label'])

  train.drop(train.loc[train['label']=='社會'].index, inplace=True)
  train.drop(train.loc[train['label']=='國際'].index, inplace=True)
  train.drop(train.loc[train['label']=='生活'].index, inplace=True)
  train.drop(train.loc[train['label']=='影劇'].index, inplace=True)
  train=train.reset_index()

  label_to_index = {
      '政治':0,
      '體育':1,   
      '財經':2, 
  }

  print(len(train))

  y_data = train.label.apply(lambda x: label_to_index[x])
  d={'label':train['label'].value_counts().index,'count':train['label'].value_counts()}
  df_cat=pd.DataFrame(data=d).reset_index(drop=True)
  print(df_cat)

  def jieba_tokenizer(text):
      words = pseg.cut(text)
      return ' '.join([word for word, flag in words if flag != 'x'])
  
  y_data = np.asarray(y_data).astype('float32')

  y_data =tensorflow.keras.utils.to_categorical(y_data)
  data=train.loc[:,use_col]
    
  data=data[:].astype(str).apply(jieba_tokenizer)
  
  label_to_index = {v: k for k, v in label_to_index.items()}
  
  acr_compare_list=[]

  num_add=0
  
  num_start_train = 0
  num_end_train = 2000

  num_start_valid = 100  
  num_end_valid = 2000
  
  num_start_test = 2000
  num_end_test = 3000

  x_test_data = data[num_start_test:num_end_test]
  y_test_data = y_data[num_start_test:num_end_test]
  
  x_not_test = data[num_start_train:num_end_valid]
  y_not_test = y_data[num_start_train:num_end_valid]

  acr_list=[]
  
  random_list=[]
  select_list=[]

  round_add_list=[]
  num_add_list=[]
  
  x_train_list=[]
  x_valid_list=[]

  wrong=[]
  correct=[]
  pro_average_list=[]
  
  wrong_second=[]
  correct_second=[]
  pro_average_list_second=[]
  
  entropy_list=[]
  entropy_max_list=[]
  entropy_list_cor=[]
  entropy_list_wrong=[]
  entropy_list_cor_max=[]
  entropy_list_wrong_max=[]

  boundup=[1.08,1.08,1.08,1.08,1.08,1.08,1.08,1.08,1.08,1.08]
  bounddown=[1.06,1.03,0.95,0.8,0.7,0.6,0.5,0.4,0.3,0.25]
  for w in range(0,19):
    
    os.environ['PYTHONHASHSEED']='1'
    tensorflow.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    print("\n")

    x_test = data[num_start_test:num_end_test]
    y_test = y_data[num_start_test:num_end_test]
    
    if (w==0):
      
      # print("base\n")
      
      x_train = data[num_start_train:num_end_train]
      y_train = y_data[num_start_train:num_end_train]
      
      x_valid = data[num_start_valid:num_end_valid]
      y_valid = y_data[num_start_valid:num_end_valid]
    
    if(w%2==1):

      # print("random\n")
      
      x_train = data[num_start_train:num_end_train+num_add]
      y_train = y_data[num_start_train:num_end_train+num_add]

      #no use
      x_valid = data[num_start_valid+num_add:num_end_valid]
      y_valid = y_data[num_start_valid+num_add:num_end_valid]

    if(w%2==0 and w!=0):
      
      # print("select\n")
      x_train = x_not_test[num_start_train:num_end_train]
      add_x_train = x_not_test[num_end_valid-num_add:num_end_valid]
      x_train = np.concatenate([x_train,add_x_train])

      y_train = y_not_test[num_start_train:num_end_train]
      add_y_train = y_not_test[num_end_valid-num_add:num_end_valid]
      y_train= np.concatenate([y_train,add_y_train])

      x_valid = x_not_test[num_start_valid:num_end_valid-num_add]
      y_valid = y_not_test[num_start_valid:num_end_valid-num_add]
    
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)
    
    tokenizer.fit_on_texts(x_train)

    # word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train,maxlen=MAX_SEQUENCE_LENGTH)

    x_valid = tokenizer.texts_to_sequences(x_valid)
    x_valid = keras.preprocessing.sequence.pad_sequences(x_valid,maxlen=MAX_SEQUENCE_LENGTH)
    
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test,maxlen=MAX_SEQUENCE_LENGTH)
    
    print("length of train:",len(x_train))
    print("length of valid:",len(x_valid))
    print("length of test",len(x_test))

    model = Sequential()
    
    model.add(Embedding(MAX_NUM_WORDS, NUM_EMBEDDING_DIM,embeddings_initializer=keras.initializers.Constant(value=0)))

    model.add(LSTM(NUM_LSTM_UNITS,kernel_initializer=initializers.glorot_normal(seed=9),recurrent_initializer=initializers.Orthogonal(gain=1.0, seed=785)))

    model.add(Dense(units=NUM_CLASSES,kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=99),activation='softmax'))
    
    print(model.summary())

    # from keras.callbacks import ModelCheckpoint
    # filepath='weights.best.hdf5'
    # checkpoint = ModelCheckpoint(filepath, monitor= 'val_accuracy', verbose=0,save_best_only=True) 
    
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

    BATCH_SIZE = batchlist[6]
    NUM_EPOCHS = 50

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(x_test,y_test),
        shuffle=False,
        verbose=0
        # callbacks=[checkpoint],
    ) 
    
    maxx=0
    
    for i in range(0,len(history.history['val_accuracy'])):
      if(history.history['val_accuracy'][i]>maxx):
        maxx=history.history['val_accuracy'][i]

    print("accuracy:",maxx)
    acr_list.append(maxx)

    # scores = model.evaluate(x_test,y_test,verbose=0)  
    # print("%s: %.1f%%" % (model.metrics_names[1], scores[1]*100))

    if(w%2==1 or w==0):
      random_list.append(round(100*maxx,3))
      
    if(w%2==0):
      select_list.append(round(100*maxx,3))

    print(acr_list)

    if(w%2==0):

      predictions = model.predict(x_valid)
      t= [label_to_index[idx] for idx in np.argmax(predictions, axis=1)]

      pro_wrong=[]
      pro_cor=[]
      pro_average=[]
      
      pro_wrong_second=[]
      pro_cor_second=[]
      pro_average_second=[]
      
      pro_entropy=[]
      pro_entropy_correct=[]
      pro_entropy_wrong=[]
      
      index=num_start_valid
      now=0
      round_add=0
    
      for tt in t:
        
        predictions_second=predictions[now].copy()
        predictions_second[np.argmax(predictions[now])]=0
        
        if(np.argmax(predictions[now])!=np.argmax(y_valid[now])):
          pro_wrong.append(100*np.max(predictions[now]))
          pro_wrong_second.append(100*np.max(predictions_second))
          pro_entropy_wrong.append(entropy(predictions[now]))

        else:
          pro_cor.append(100*np.max(predictions[now]))
          pro_cor_second.append(100*np.max(predictions_second))
          pro_entropy_correct.append(entropy(predictions[now]))

        pro_average.append(100*np.max(predictions[now]))
        pro_average_second.append(100*np.max(predictions_second))
        pro_entropy.append(entropy(predictions[now]))

        if(w<5):
          if(100*np.max(predictions[now])-100*np.max(predictions_second)<1+w):
          # if(100*np.max(predictions[now])>37 and 100*np.max(predictions[now])<40):
          # if((100*np.max(predictions[now])-100*np.max(predictions_second)<1+(w/2))  
            x_not_test.loc[num_end_valid] = x_not_test[index] 
            x_not_test = x_not_test.drop(x_not_test.index[index])
            x_not_test = x_not_test.reset_index(drop=True)

            y_not_test = np.insert(y_not_test,num_end_valid,values=y_not_test[index],axis=0)
            y_not_test = np.delete(y_not_test,(index),axis=0)

            num_add+=1
            round_add+=1
        
        # if(w==2 or w==4):
        #   # if((100*np.max(predictions[now])-100*np.max(predictions_second)<1+(w/2))
        #   if(100*np.max(predictions[now])-100*np.max(predictions_second)<1+w):
        #   # if(100*np.max(predictions[now])-100*np.max(predictions_second)<2+w/2 and 100*np.max(predictions[now])-100*np.max(predictions_second)>1):
        #     x_not_test.loc[num_end_valid] = x_not_test[index] 
        #     x_not_test = x_not_test.drop(x_not_test.index[index])
        #     x_not_test = x_not_test.reset_index(drop=True)

        #     y_not_test = np.insert(y_not_test,num_end_valid,values=y_not_test[index],axis=0)
        #     y_not_test = np.delete(y_not_test,(index),axis=0)

        #     num_add+=1
        #     round_add+=1
        # if(w>3):
        #   # if((100*np.max(predictions[now])-100*np.max(predictions_second)<1+(w/2))
        #   # if(100*np.max(predictions[now])>75 and 100*np.max(predictions[now])<80):
        #   # if(100*np.max(predictions[now])-100*np.max(predictions_second)<1+(w/2)): 
        #   if(100*np.max(predictions[now])<65 and 100*np.max(predictions[now])>60): 
        #     x_not_test.loc[num_end_valid] = x_not_test[index] 
        #     x_not_test = x_not_test.drop(x_not_test.index[index])
        #     x_not_test = x_not_test.reset_index(drop=True)

        #     y_not_test = np.insert(y_not_test,num_end_valid,values=y_not_test[index],axis=0)
        #     y_not_test = np.delete(y_not_test,(index),axis=0)

        #     num_add+=1
        #     round_add+=1
        # if(w>7 and w<13):
        #   temp=w-8
        #   # temp=temp/60
        #   if(entropy(predictions[now])>0.6-temp*0.03 and entropy(predictions[now])<0.8):
        #     x_not_test.loc[num_end_valid] = x_not_test[index] 
        #     x_not_test = x_not_test.drop(x_not_test.index[index])
        #     x_not_test = x_not_test.reset_index(drop=True)

        #     y_not_test = np.insert(y_not_test,num_end_valid,values=y_not_test[index],axis=0)
        #     y_not_test = np.delete(y_not_test,(index),axis=0)

        #     num_add+=1
        #     round_add+=1
        if(w>5):
          temp=int(w/2)
          # print(temp)
          if(entropy(predictions[now])> bounddown[temp] and entropy(predictions[now])<boundup[temp]):
            x_not_test.loc[num_end_valid] = x_not_test[index] 
            x_not_test = x_not_test.drop(x_not_test.index[index])
            x_not_test = x_not_test.reset_index(drop=True)

            y_not_test = np.insert(y_not_test,num_end_valid,values=y_not_test[index],axis=0)
            y_not_test = np.delete(y_not_test,(index),axis=0)

            num_add+=1
            round_add+=1

        now+=1
        index+=1

      round_add_list.append(round_add)
      num_add_list.append(num_add)
      
      correct.append(sum(pro_cor)/len(pro_cor))
      wrong.append(sum(pro_wrong)/len(pro_wrong))
      pro_average_list.append(sum(pro_average)/len(pro_average))
      
      correct_second.append(sum(pro_cor_second)/len(pro_cor_second))
      wrong_second.append(sum(pro_wrong_second)/len(pro_wrong_second))
      pro_average_list_second.append(sum(pro_average_second)/len(pro_average_second))

      entropy_list.append(sum(pro_entropy)/len(pro_entropy))
      entropy_max_list.append(max(pro_entropy))
      entropy_list_cor_max.append(max(pro_entropy_correct))
      entropy_list_cor.append(sum(pro_entropy_correct)/len(pro_entropy_correct))
      entropy_list_wrong_max.append(max(pro_entropy_wrong))
      entropy_list_wrong.append(sum(pro_entropy_wrong)/len(pro_entropy_wrong))

  print("random:",random_list)
  print("select:",select_list)

  print("num_add",num_add_list)
  print("round_add:",round_add_list)
  
  print('cor:',correct)
  print("wrong:",wrong)
  print("pro_average:",pro_average_list)
  
  print('cor_second:',correct_second)
  print("wrong_second:",wrong_second)
  print("pro_average_second:",pro_average_list_second)

  print("entropy_list:",entropy_list)
  print("entropy_max_list:",entropy_max_list)
  print("entropy_list_cor:",entropy_list_cor)
  print("entropy_list_cor_max:",entropy_list_cor_max)
  print("entropy_list_wrong:",entropy_list_wrong)
  print("entropy_list_wrong_max:",entropy_list_wrong_max)
