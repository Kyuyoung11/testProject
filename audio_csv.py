import csv
from collections import Counter

def most_common_top_1(candidates):
    """후보자 중 가장 많은 득표자를 뽑는다(단, 동점발생 시 리스트 왼쪽을 우선한다)"""
    assert isinstance(candidates, list), 'Must be a list type'
    if len(candidates) == 0: return None
    return Counter(candidates).most_common(n=1)[0][0]


f = open('audio_sentiment.csv','r',encoding="UTF-8")
rdr = csv.reader(f)

for line in rdr:
    sentiment_line = []
    sentiment_line.append(line[3])
    sentiment_line.append(line[5])
    sentiment_line.append(line[7])
    sentiment_line.append(line[9])
    sentiment_line.append(line[11])
    print(sentiment_line)
    print(most_common_top_1(sentiment_line))

f.close()