import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding
import pickle
import xgboost as xgb
from xgboost import plot_importance
from xgboost import XGBClassifier
import csv
from collections import Counter

mslen = 22050

data = []

max_fs = 0
labels = []

emotions = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
label_max = 1900
label = 0

path_data = ["4_wav", "5_wav"]
file_name = ['audio_sentiment.csv', 'audio_sentiment5.csv']


# file_name = ['audio_sentiment_test.csv']


def most_common_top_1(candidates):
    """후보자 중 가장 많은 득표자를 뽑는다(단, 동점발생 시 리스트 왼쪽을 우선한다)"""
    assert isinstance(candidates, list), 'Must be a list type'
    if len(candidates) == 0: return None
    return Counter(candidates).most_common(n=1)[0][0]


i = 0
feature_all = np.array([])
for a in range(0, len(file_name)):
    directories = os.listdir(path_data[a])

    print(directories)
    f = open(file_name[a], 'r', encoding='utf-8-sig')
    rdr = csv.reader(f)

    for line in rdr:
        if (line[0] + ".wav") not in directories: continue
        print(line[0])
        file_path = path_data[a] + "/" + line[0] + ".wav"

        X, sr = librosa.load(file_path, sr=None)

        sentiment_line = []
        sentiment_line.append(line[3])
        sentiment_line.append(line[5])
        sentiment_line.append(line[7])
        sentiment_line.append(line[9])
        sentiment_line.append(line[11])
        most_sentiment = most_common_top_1(sentiment_line)
        print(most_sentiment)
        if most_sentiment == "Neutral":
            label = 0
        elif most_sentiment == "Happiness":
            label = 1
        elif most_sentiment == "Angry":
            label = 2
        elif most_sentiment == "Sadness":
            label = 3
        elif most_sentiment == "Disgust":
            label = 4
        elif most_sentiment == "Surprise":
            label = 5
        else:
            label = 6

        if (labels.count(label) > label_max):
            continue
        else:
            labels.append(label)

        stft = np.abs(librosa.stft(X))

        ############# EXTRACTING AUDIO FEATURES #############
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

        mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)

        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

        if (i == 0):
            feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        else:
            features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
            feature_all = np.vstack([feature_all, features])

        i += 1

    f.close()

# 추가로 수집한 joy
directories = os.listdir("joy_wav")
print(directories)

for a in directories:
    labels.append(1)
    file_path = "joy_wav/" + a

    X, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(X))

    ############# EXTRACTING AUDIO FEATURES #############
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

    if (i == 0):
        feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    else:
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        feature_all = np.vstack([feature_all, features])

# 추가로 수집한 surprise
directories = os.listdir("sur_wav")
print(directories)

for a in directories:
    labels.append(5)
    file_path = "sur_wav/" + a

    X, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(X))

    ############# EXTRACTING AUDIO FEATURES #############
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

    if (i == 0):
        feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    else:
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        feature_all = np.vstack([feature_all, features])

'''
#추가로 수집한 disgust
directories = os.listdir("result_dis")
print(directories)

for a in directories:
    labels.append(4)
    file_path = "result_dis/" + a

    X, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(X))

    ############# EXTRACTING AUDIO FEATURES #############
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

    if (i == 0):
        feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    else:
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        feature_all = np.vstack([feature_all, features])

#추가로 수집한 none
directories = os.listdir("result_non")
print(directories)

for a in directories:
    labels.append(0)
    file_path = "result_non/" + a

    X, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(X))

    ############# EXTRACTING AUDIO FEATURES #############
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

    if (i == 0):
        feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    else:
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        feature_all = np.vstack([feature_all, features])

#추가로 수집한 fear
directories = os.listdir("result_fea")
print(directories)

for a in directories:
    labels.append(6)
    file_path = "result_fea/" + a

    X, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(X))

    ############# EXTRACTING AUDIO FEATURES #############
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)

    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

    if (i == 0):
        feature_all = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    else:
        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        feature_all = np.vstack([feature_all, features])
'''

for i in range(0, len(emotions)):
    print(emotions[i] + " : " + str(labels.count(i)))

from copy import deepcopy

y = deepcopy(labels)
for i in range(len(y)):
    y[i] = int(y[i])

n_labels = len(y)
n_unique_labels = len(np.unique(y))
one_hot_encode = np.zeros((n_labels, n_unique_labels))
f = np.arange(n_labels)
for i in range(len(f)):
    one_hot_encode[f[i], y[i] - 1] = 1
print(feature_all)
print(one_hot_encode)

X_train, X_test, y_train, y_test = train_test_split(feature_all, one_hot_encode, test_size=0.3, shuffle=True,
                                                    random_state=20)

'''
########################### MODEL 1 ###########################
model = Sequential()

#model.add(Dense(X_train.shape[1],input_dim =X_train.shape[1],init='normal',activation ='relu'))
model.add(Dense(X_train.shape[1],input_dim =X_train.shape[1], activation ='relu'))

model.add(Dense(400,activation ='relu'))

model.add(Dropout(0.2))

model.add(Dense(200,activation ='relu'))

model.add(Dropout(0.2))

model.add(Dense(100,activation ='relu'))

model.add(Dropout(0.2))
print("y_trainshape" + str(y_train.shape[1]))
model.add(Dense(y_train.shape[1],activation ='softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model.fit(X_train,y_train, epochs=500, batch_size = 64,verbose=1)


model.evaluate(X_test,y_test)
model.summary()

mlp_model = model.to_json()
with open('mlp_model_relu_adadelta.json','w') as j:
    j.write(mlp_model)
model.save_weights("mlp_relu_adadelta_model.h5")

y_pred_model1 = model.predict(X_test)
y2 = np.argmax(y_pred_model1,axis=1)
y_test2 = np.argmax(y_test , axis = 1)

count = 0
for i in range(y2.shape[0]):
    if y2[i] == y_test2[i]:
        count+=1

print('Accuracy for model 1 : ' + str((count / y2.shape[0]) * 100))

########################### MODEL 2 ###########################
model2 = Sequential()

model2.add(Dense(X_train.shape[1],input_dim =X_train.shape[1],activation ='relu'))

model2.add(Dense(400,activation ='tanh'))

model2.add(Dropout(0.2))

model2.add(Dense(200,activation ='tanh'))

model2.add(Dropout(0.2))

model2.add(Dense(100,activation ='sigmoid'))

model2.add(Dropout(0.2))

model2.add(Dense(y_train.shape[1],activation ='softmax'))

model2.compile(loss = 'categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

model2.fit(X_train,y_train,epochs=500,batch_size = 64,verbose=1)

model2.evaluate(X_test, y_test)


mlp_model2 = model2.to_json()
with open('mlp_model_tanh_adadelta.json','w') as j:
    j.write(mlp_model2)
model2.save_weights("mlp_tanh_adadelta_model.h5")


y_pred_model2 = model2.predict(X_test)
y22 = np.argmax(y_pred_model2,axis=1)
y_test22 = np.argmax(y_test , axis = 1)

count = 0
for i in range(y22.shape[0]):
    if y22[i] == y_test22[i]:
        count+=1

print('Accuracy for model 2 : ' + str((count / y22.shape[0]) * 100))
model2.summary()

'''

X_train2, X_test2, y_train2, y_test2 = train_test_split(feature_all, y, test_size=0.3, random_state=30)
eval_s = [(X_train2, y_train2), (X_test2, y_test2)]
########################### MODEL 3 ###########################

model3 = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=4)
model3.fit(X_train2, y_train2, eval_set=eval_s)
model3.evals_result()
score = cross_val_score(model3, X_train2, y_train2, cv=5)
y_pred3 = model3.predict(X_test2)

count = 0
for i in range(y_pred3.shape[0]):
    if y_pred3[i] == y_test2[i]:
        count += 1

print('Accuracy for model 3 : ' + str((count / y_pred3.shape[0]) * 100))

# 파일명
filename = 'audio_model/xgb_model5004.model'

# 모델 저장
pickle.dump(model3, open(filename, 'wb'))

model4 = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=8)
model4.fit(X_train2, y_train2, eval_set=eval_s)
model4.evals_result()
score = cross_val_score(model4, X_train2, y_train2, cv=5)
y_pred4 = model4.predict(X_test2)

count = 0
for i in range(y_pred4.shape[0]):
    if y_pred4[i] == y_test2[i]:
        count += 1

print('Accuracy for model 3 : ' + str((count / y_pred4.shape[0]) * 100))

# 파일명
filename = 'audio_model/xgb_model5008.model'

# 모델 저장
pickle.dump(model4, open(filename, 'wb'))

# 파일명
filename = 'audio_model/xgb_model5004.model'

# 모델 저장
pickle.dump(model3, open(filename, 'wb'))

model5 = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4)
model5.fit(X_train2, y_train2, eval_set=eval_s)
model5.evals_result()
score = cross_val_score(model5, X_train2, y_train2, cv=5)
y_pred5 = model5.predict(X_test2)

count = 0
for i in range(y_pred5.shape[0]):
    if y_pred5[i] == y_test2[i]:
        count += 1

print('Accuracy for model 3 : ' + str((count / y_pred5.shape[0]) * 100))

# 파일명
filename = 'audio_model/xgb_model3004.model'

# 모델 저장
pickle.dump(model5, open(filename, 'wb'))

'''
########################### TESTING ###########################
test_file_path = "5ebd3dd7c38c123b9ec6deba.wav"
X,sr = librosa.load(test_file_path, sr = None)
stft = np.abs(librosa.stft(X))

############# EXTRACTING AUDIO FEATURES #############
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40),axis=1)

chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)

mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)

contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr,fmin=0.5*sr* 2**(-6)).T,axis=0)

tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sr*2).T,axis=0)

features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])

feature_all = np.vstack([feature_all,features])




x_chunk = np.array(features)
x_chunk = x_chunk.reshape(1,np.shape(x_chunk)[0])
y_chunk_model1 = model.predict(x_chunk)
print(y_chunk_model1)
index = np.argmax(y_chunk_model1)
print(index)
print('Emotion:',emotions[index])
'''