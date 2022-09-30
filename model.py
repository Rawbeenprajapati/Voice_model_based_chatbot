#### Extracting MFCC's For every audio file
import pandas as pd
import os
import librosa

audio_dataset_path='college_dataset/audio'

metadata=pd.read_csv('college_dataset/metadata/college_dataset.csv')

###pip install nltk,tensorflow,keras,pickle,re,mysql.connector,numpy
from turtle import goto
import nltk
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# stop_word =stopwords.words('english')
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import re
from keras.models import load_model
model = load_model('C:/KhecVoiceBot/NEWAIASSISTANTBYBACH/chatbot_model.h5')
import json
import random
import mysql.connector

#Creating GUI with tkinter
# import tkinter
from tkinter import *
base = Tk()
name_var=StringVar()

# stop_word=["a","the","an","c'mon","co","co.","t's","un","unto","v","viz","vs","a","b","c","d","e","f","g","h","j","l","m","n","o","p","q","r","s","t","u","uucp","w","x","y","z"]
#--------
intents = json.loads(open('C:/KhecVoiceBot/NEWAIASSISTANTBYBACH/intents.json').read())
words = pickle.load(open('C:/KhecVoiceBot/NEWAIASSISTANTBYBACH/words.pkl','rb'))
classes = pickle.load(open('C:/KhecVoiceBot/NEWAIASSISTANTBYBACH/classes.pkl','rb'))
bot_name = "khec_Bot"

#--------

def cleaning(sentence):
    sentence= sentence.lower()
    sentence = re.sub(r"i'm","i am",sentence)
    sentence = re.sub(r"he's","he is",sentence)	
    sentence = re.sub(r"she's","she is",sentence)	
    sentence = re.sub(r"that's","that is",sentence)
    sentence = re.sub(r"what's","what is",sentence)	
    sentence = re.sub(r"where's","where is",sentence)		
    sentence = re.sub(r"\'ll","will",sentence)	
    sentence = re.sub(r"\'ve","have",sentence)	
    sentence = re.sub(r"\'re","are",sentence)	
    sentence = re.sub(r"\'d","will",sentence)	
    sentence = re.sub(r"won't","will not",sentence)	 
    sentence = re.sub(r"can't","cannot",sentence)	
    sentence = re.sub(r"[-()\"#/@;:<>=|.?,]","",sentence)
    sentence_words = nltk.word_tokenize(sentence)

    # filter_stopword=[t for t in sentence_words if t not in stop_word]
    filter_word = list(filter(lambda x: x in classes or words, sentence_words))
    print("###########_______###############---------------"+str(filter_word)+"______________##########################")
    return filter_word
#--------

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words=cleaning(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"+str(res)+"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    # ERROR_THRESHOLD = 0.20

    results = [[i,r] for i,r in enumerate(res)]     #=> results=[[i,r],[i,r]....]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)  # x=[i,r]

    print("========="+str(results[0])+"==============================================")
    print("========="+str(results[1])+"==============================================")
    print("========="+str(results[3])+"==============================================")

    return_list = []
    # print("****-----------------------"+classes[86]+"-----------------------********")
    return_list.append({"intent": classes[results[0][0]], "probability": str(results[0][1])})
    print("++++++++++++++++++++"+str(return_list)+"++++++++++++++++++++++++++") 
    print("++++++++++++++++++++"+str({"intent": classes[results[1][0]], "probability": str(results[1][1])})+"++++++++++++++++++++++++++") 
    print("++++++++++++++++++++"+str({"intent": classes[results[3][0]], "probability": str(results[3][1])})+"++++++++++++++++++++++++++") 

    # for r in results[0]:
    #     return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    # print("++++++++++++++++++++"+str(return_list)+"++++++++++++++++++++++++++")    
    return return_list
    
def getResponse(ints, intents_json,tagging=False):
    if tagging == True:
        tag = ints
    else:
        tag = ints[0]['intent']
    
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):  
    if text in classes:
        res = getResponse(text,intents,tagging=True)
        print("This is my response==================>"+str(res))
        return res

    ints = predict_class(text, model)  
    prob=float(ints[0]['probability']) #filtering the highest
    print(type(prob))
    if prob > 0.77:
        res = getResponse(ints, intents)
    else:
        res="I can't get it,Can you please reformulae it and say or type again?"
        # res="Hey, I'm only a bot, I need things simple.Could you please place query more detailly or Exclude slang words?,Thank you"  

        # mydb = mysql.connector.connect(host="localhost", user="root", passwd="",database="chatbot")

        # mycursor = mydb.cursor()
        # mycursor.execute(f"select * from `new_query` where `Query`='{text}'")
        # lst_cursor=list(mycursor)
        # l=len(lst_cursor)
        

        # if l!=0:
        #     print(lst_cursor[0][2])
        #     val=lst_cursor[0][2]+1
        #     print("updating...")
        #     print(val)
        #     mycursor = mydb.cursor()
        #     mycursor.execute(f"UPDATE `new_query` SET `freq`= {val} where `Query`='{text}'")
        #     mydb.commit()
        # else:
        #     print("inserting...")
        #     mycursor = mydb.cursor()
        #     # SELECT * FROM `new_query` WHERE `query`='xyz'
        #     query = f"INSERT INTO `new_query` (`Query`) VALUES ('{text}')"
        #     mycursor.execute(query)
        #     mydb.commit()
        # mydb.close()    
    return res
        # ////////////////////////////////////////////////////////////////////////////////////////





#metadata.tail()
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

import numpy as np
from tqdm import tqdm
### Now we iterate through every audio file and extract features 
### using Mel-Frequency Cepstral Coefficients
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
   #print(file_name)
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()

### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

### Label Encoding
###y=np.array(pd.get_dummies(y))
### Label Encoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))


### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
num_labels=y.shape[1]
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
# model.add(Conv2D(100, kernel_size=3, activation='relu', input_shape=(40,)))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 500
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification_collegeDataset.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)



test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1]*100)
import pyaudio
import wave

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
seconds = 6
filename = "record/output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording.....')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

filename="record/output.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)
predicted_label=np.argmax(model.predict(mfccs_scaled_features),axis=1)
#predicted_label=model.predict_classes(mfccs_scaled_features)
print(predicted_label)
prediction_class = labelencoder.inverse_transform(predicted_label) 
#print(prediction_class[0])
res = chatbot_response(prediction_class[0])
print(res)
