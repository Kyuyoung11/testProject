import numpy as np
import librosa
import pickle

import wave

import torch
from transformers import AutoTokenizer
import mymodel
import re


class audioClassification():
    def __init__(self):
        self.labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]

        # 음성 모델 파일명
        self.filename = 'xgb_model.model'

        # 음성 모델 불러오기
        self.loaded_model = pickle.load(open(self.filename, 'rb'))

        # 텍스트 모델 초기값
        self.none_words = ["안싫", "안 싫", "안무서", "안놀람", "안놀랐", "안행복", "안기뻐", "안빡", "안우울", "안짜증", "안깜짝", "안무섭"]
        self.pass_words = ["안좋", "안 좋"]
        self.senti_loss = [5.0, 4.0, 6.5, 6.5, 9.0, 8.0]
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
        # GPU 사용
        self.device = torch.device("cuda")

        # 텍스트 모델 불러오기
        self.model = mymodel.HwangariSentimentModel.from_pretrained("Kyuyoung11/haremotions-v2").to(self.device)

    def classify(self, audio_path, text):

        ########################### TESTING ###########################
        # test_file_path = "5_wav/5f05fb0bb140144dfcff0184.wav"
        X, sr = librosa.load(audio_path, sr=None)
        stft = np.abs(librosa.stft(X))

        ############# EXTRACTING AUDIO FEATURES #############
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40), axis=1)

        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

        mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)

        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr, fmin=0.5 * sr * 2 ** (-6)).T, axis=0)

        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr * 2).T, axis=0)

        features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        x_chunk = np.array(features)
        x_chunk = x_chunk.reshape(1, np.shape(x_chunk)[0])
        # y_chunk_model1 = self.loaded_model.predict(x_chunk)
        y_chunk_model1_proba = self.loaded_model.predict_proba(x_chunk)
        index = np.argmax(y_chunk_model1_proba)

        print("-----<Accuracy>------")
        for proba in range(0, len(y_chunk_model1_proba[0])):
            print(self.labels[proba] + " : " + str(y_chunk_model1_proba[0][proba]))

        print('\nEmotion:', self.labels[int(index)])
        print("--------------------")

        # enc = tokenizer.encode_plus(text)
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        # print(tokenizer.tokenize(tokenizer.decode(enc["input_ids"])))

        self.model.eval()

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        output = self.model(input_ids.to(self.device), attention_mask.to(self.device))[0]
        _, prediction = torch.max(output, 1)

        label_loss_str = str(output).split(",")

        label_loss = [float(x.strip().replace(']', '')) for x in label_loss_str[1:7]]

        print(f'Review text : {text}')

        pre_result = int(re.findall("\d+", str(prediction))[0])

        # 손실함수 값이 senti_loss 값보다 높아야 해당 감정으로 분류
        result = 0
        if label_loss[pre_result - 1] >= self.senti_loss[pre_result - 1]:
            result = pre_result

        # 안이 들어간 말로 결과가 나왔을 경우 가장 큰 값을 무시함 or 아예 무감정으로 분류되도록 함
        for i in self.none_words:
            if i in text:
                result = 0
        for j in self.pass_words:
            if j in text:
                label_loss[pre_result - 1] = 0
                result = label_loss.index(max(label_loss)) + 1

        print(f'Sentiment : {self.labels[result]}')

        print("\n<감정 별 손실 함수 값>")
        for i in range(0, 6):
            print(self.labels[i + 1], ":", label_loss[i])

        if (index == 0):
            print("b")
            total_result = -1
        elif (index == result):
            print("a")
            total_result = result

        else:
            text_score = []
            audio_score = []
            total_score = []
            for i in range(0, len(label_loss)):
                text_score.append(label_loss[i] / (sum(label_loss) + 10))
                audio_score.append(y_chunk_model1_proba[0][i + 1] - 0.35)

            for i in range(0, len(audio_score)):
                total_score.append(float(audio_score[i]) + float(text_score[i]))
            print(total_score)

            total_result = total_score.index(max(total_score))
        print("Result : ", self.labels[total_result + 1])
        return total_result