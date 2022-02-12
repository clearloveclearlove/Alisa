import pandas as pd
from sklearn.model_selection import train_test_split
import random
random.seed(2021)

cover_df=pd.read_csv('./rnn_steg_data/tweet/flc/cover.csv')
# stega_df=pd.read_csv('./data/tweet/vlc/1bit.csv')
stega_df=pd.read_csv('./rnn_steg_data/tweet/flc/3bit.csv')

cover=cover_df['sentence'].values.tolist()
stega=stega_df['sentence'].values.tolist()
cover.extend(stega)

cover_label=[0]*10000
stega_label=[1]*10000

cover_label.extend(stega_label)

dic={'sentence':cover,'label':cover_label}

df=pd.DataFrame(dic,columns=['sentence','label'])

train,test=train_test_split(df,test_size=0.3,stratify=df['label'],random_state=2021)
train,val=train_test_split(train,test_size=0.1,stratify=train['label'],random_state=2021)

pretrain_data=train[train['label']==0]

train.to_csv('./data/train.csv')
val.to_csv('./data/val.csv')
test.to_csv('./data/test.csv')
