import re
import os
import csv
import re
from collections import Counter
import shutil



def cleanText(readData):
    # 텍스트에 포함되어 있는 특수 문자 제거

    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)

    return text


def read_csv():
    f2 = open('text_data_joy.csv', 'w', encoding='utf-8-sig', newline='')
    wr = csv.writer(f2)
    wr.writerow(["text","index"])
    f = open('text_data.csv', 'r', encoding='utf-8-sig')
    rdr = csv.reader(f)

    for line in rdr:


        if (line[1] == "혐오"):
            line[0] = cleanText(line[0]).strip()
            wr.writerow([line[0], 1])
    f2.close()
    f.close()

def read_csv2():
    f2 = open('text_data_joy.csv', 'a', encoding='utf-8-sig', newline='')
    wr = csv.writer(f2)
    f = open('text_data2.csv', 'r', encoding='utf-8-sig')
    rdr = csv.reader(f)


    for line in rdr:

        if (line[2] == "행복"):
            line[1] = cleanText(line[1]).strip()
            wr.writerow([line[1], 1])

    f2.close()
    f.close()

read_csv2()