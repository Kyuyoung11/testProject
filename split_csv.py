
import numpy as np
import os
import csv
import re
from collections import Counter
import shutil


mslen = 22050

data = []

max_fs = 0
labels = []



emotions = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]


file_name = ['audio_sentiment.csv', 'audio_sentiment5.csv', 'audio_sentiment7.csv']
file_name2 = 'audio_sentiment6.csv'
test_file = []


def cleanText(readData):
    # 텍스트에 포함되어 있는 특수 문자 제거

    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)

    return text


def most_common_top_1(candidates):
    """후보자 중 가장 많은 득표자를 뽑는다(단, 동점발생 시 리스트 왼쪽을 우선한다)"""
    assert isinstance(candidates, list), 'Must be a list type'
    if len(candidates) == 0: return None
    return Counter(candidates).most_common(n=1)[0][0]


def split1():
    i = 0
    label = 0

    f2 = open('text_fromaudio.csv', 'w', encoding='utf-8-sig', newline='')
    wr = csv.writer(f2)

    wr.writerow(['text', 'index'])


    for a in range(0, len(file_name)):
        f = open(file_name[a], 'r', encoding='utf-8-sig')
        rdr = csv.reader(f)

        for line in rdr:
            #print(line[0])

            #X, sr = librosa.load(file_path, sr=None)

            sentiment_line = []
            sentiment_line.append(line[3])
            sentiment_line.append(line[5])
            sentiment_line.append(line[7])
            sentiment_line.append(line[9])
            sentiment_line.append(line[11])
            most_sentiment = most_common_top_1(sentiment_line)
            if most_sentiment.lower() == "neutral":
                label = 0
            elif most_sentiment.lower() == "happiness":

                label = 1
            elif most_sentiment.lower() == "angry":
                label = 2
            elif most_sentiment.lower() == "sadness":
                label = 3
            elif most_sentiment.lower() == "disgust":
                label = 4
            elif most_sentiment.lower() == "surprise":
                label = 5
            elif most_sentiment.lower() == "fear":
                label = 6




            labels.append(label)

            text = cleanText(line[1])
            wr.writerow([text, label])



            i += 1

        f.close()

    f2.close()



    for i in range(0, len(emotions)):
        print(emotions[i] + " : " + str(labels.count(i)))

def split2():
    i = 0
    label = 0

    f2 = open('text_fromaudio2.csv', 'w', encoding='utf-8-sig', newline='')
    wr = csv.writer(f2)

    wr.writerow(['text', 'index'])


    f = open(file_name2, 'r', encoding='utf-8-sig')
    rdr = csv.reader(f)

    for line in rdr:

        if line[5] == "무감정":
            label = 0
        elif line[5] == "기쁨":
            label = 1
        elif line[5] == "분노":
            label = 2
        elif line[5] == "슬픔":
            label = 3
        elif line[5] == "혐오":
            label = 4
        elif line[5] == "놀람":
            label = 5
        elif line[5] == "무서움":
            label = 6


        labels.append(label)

        text = cleanText(line[7])
        wr.writerow([text, label])



        i += 1

    f.close()

    f2.close()



    for i in range(0, len(emotions)):
        print(emotions[i] + " : " + str(labels.count(i)))










split2()