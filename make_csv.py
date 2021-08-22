import os
import csv
import re
from collections import Counter


# file_name = ["regex_result_annoy.csv", "regex_result_disgust.csv", "regex_result_fear.csv", "regex_result_joy.csv",
#                  "regex_result_sad.csv", "regex_result_surprise.csv", "text_data_annoy.csv", "text_data_disgust.csv",
#                  "text_data_fear.csv", "text_data_joy.csv", "text_data_neutral.csv", "text_data_sad.csv",
#                  "text_data_surprise.csv", "text_fromaudio.csv", "text_fromaudio2.csv"]
file_name = ["text_data_annoy.csv", "text_data_disgust.csv",
            "text_data_fear.csv", "text_data_joy.csv", "text_data_neutral.csv", "text_data_sad.csv",
            "text_data_surprise.csv", "text_fromaudio.csv", "text_fromaudio2.csv"]

def check_csv():


    emotions = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
    label_count = [0] * len(emotions)

    for a in file_name:
        print(a)
        f = open(a, 'r', encoding='utf-8-sig')
        rdr = csv.reader(f)
        for line in rdr:
            if (line[0] == "text"): continue

            label_count[int(line[1])] += 1

    print(label_count)

    f.close()

def make_csv():

    label_max = 9000
    none_max = 12000
    train_max = label_max * 0.7
    test_max = label_max * 0.3
    train_none = none_max * 0.7
    test_none = none_max * 0.3
    emotions = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
    label_count = [0] * len(emotions)
    f2 = open('result_sentiment_' + str(label_max) +'_train_sur.csv', 'w', encoding='utf-8-sig', newline='')
    wr2 = csv.writer(f2)
    wr2.writerow(["text", "index"])

    f3 = open('result_sentiment_' + str(label_max) + '_test_sur.csv', 'w', encoding='utf-8-sig', newline='')
    wr3 = csv.writer(f3)
    wr3.writerow(["text", "index"])
    for a in file_name:
        print(a)
        f = open(a, 'r', encoding='utf-8-sig')
        rdr = csv.reader(f)
        for line in rdr:
            if (line[0] == "text"): continue
            if (int(line[1]) == 0 or int(line[1]) == 5):
                if (label_count[int(line[1])] >= none_max):
                    continue
                elif (label_count[int(line[1])] < train_none):
                    wr2.writerow([line[0], line[1]])
                    label_count[int(line[1])] += 1
                elif (label_count[int(line[1])] - train_none < test_none):
                    wr3.writerow([line[0], line[1]])
                    label_count[int(line[1])] += 1

            else:
                if (label_count[int(line[1])] >= label_max): continue
                elif (label_count[int(line[1])] < train_max) :
                    wr2.writerow([line[0], line[1]])
                    label_count[int(line[1])] += 1
                elif (label_count[int(line[1])]-train_max < test_max) :
                    wr3.writerow([line[0], line[1]])
                    label_count[int(line[1])] += 1

    print(label_count)

    f.close()
    f2.close()
    f3.close()



make_csv()
