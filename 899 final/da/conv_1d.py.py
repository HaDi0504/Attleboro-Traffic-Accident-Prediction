#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import random
import json
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling1D,AveragePooling1D,LeakyReLU,concatenate,Input,Conv1D,Activation,BatchNormalization,MaxPooling1D,Dense,Dropout,Flatten,LSTM
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.optimizers import Adam


def read_csv(filepath):
    df = pd.read_csv(filepath)
    df_data = df.iloc[:,2:]
    return df_data

def reduce_df_data(df_data):
    result = pd.DataFrame(df_data['label'].value_counts())["label"].tolist()
    """
    暂不处理

    """
    return df_data
def related_str_table(class_le,name):
    related_date = {}
    for cl in class_le.classes_:
        related_date.update({cl:int(class_le.transform([cl])[0])})
    with open("data/" + name + ".txt","w") as f:
        json.dump(related_date,f,indent=4)
def datasets_stand(df_data):
    scaler  =  StandardScaler()
    df_data_stand = scaler.fit_transform(df_data.iloc[:,:-1])
    return df_data_stand

def slove_df_data(df_data):  
    print ("split".center(10,"-"))
    df_data["DATE"]  = pd.to_datetime(df_data["DATE"],infer_datetime_format=False)
    df_data["DATE"] = df_data["DATE"].apply(lambda x:int(str(x).replace("-","").split(" ")[0]))
    df_data["DATE"] = df_data["DATE"]/10000000
    

 
    print ("split".center(10,"-"))

    class_le = LabelEncoder()
    df_data['Month'] = class_le.fit_transform(df_data['Month'])
    related_str_table(class_le,"Month")
    
    df_data['Day of week'] = class_le.fit_transform(df_data['Day of week'])
    related_str_table(class_le,"Day of week")

    df_data['Crash Time'] = df_data['Crash Time'].apply(lambda x:float(x.replace(":",".")))
    related_str_table(class_le,"Crash Time")

    df_data['Road Surface Condition'] = class_le.fit_transform(df_data['Road Surface Condition'])
    related_str_table(class_le,"Road Surface Condition")


    df_data['Ambient Light'] = class_le.fit_transform(df_data['Ambient Light'])
    related_str_table(class_le,"Ambient Light")

    df_data['Weather Condition'] = class_le.fit_transform(df_data['Weather Condition'])
    related_str_table(class_le,"Weather Condition")

    #df_data['label'] = class_le.fit_transform(df_data['label'])
    #df_data
    # 数据标准化
    df_data_stand = datasets_stand(df_data)
   
    return df_data



# 定义神经网络
def baseline_model():
 
    model = Sequential()
    model.add(Conv1D(filters=5, kernel_size=3,strides=1, input_shape=(11, 1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling1D(pool_size=2,strides=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=5, kernel_size=3,strides=1))
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=2,strides=2))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=5, kernel_size=3,strides=1))
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=2,strides=2))
    model.add(Conv1D(filters=5, kernel_size=3,strides=1))
    model.add(LeakyReLU())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1,activation='sigmoid'))
    print(model.summary()) # 显示网络结构
    Adam_optimizer = Adam(lr=1e-4)
    model.compile(loss="binary_crossentropy",optimizer="adam", metrics=['accuracy'])
    return model





def lstm_model():
    model = Sequential()
    model.add(LSTM(128,input_shape=(13, 1),activation="relu",return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(128,activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(100,activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(92))
    Adam_optimizer = Adam(lr=1e-6)
    #model.compile(loss="SparseCategoricalCrossentropy",optimizer='adam', metrics=['accuracy'])
    model.compile(loss="mse",optimizer='adam', metrics=['accuracy'])
    return model



def train_model(df_data_stand):
    X = np.expand_dims(df_data_stand.values[:,:-3], axis=2)
    Y = np.expand_dims(df_data_stand.values[:, -1], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=0)

    print (X_train.shape,Y_train.shape)

    X_train = X_train.astype('float64')
    Y_train = Y_train.astype('float64')
    #model = lstm_model()
    model = baseline_model()
    history = model.fit(X_train, Y_train, batch_size=100,epochs=50)







def main():
    filepath = "fff.csv"
    df_data = read_csv(filepath)

    reduce_df_data(df_data)


    
    print (df_data.shape)
    df_data = slove_df_data(df_data)
    print (df_data)
    train_model(df_data)
main()
