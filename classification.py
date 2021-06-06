import pandas as pd
import torch
from transformers import AutoTokenizer, ElectraForSequenceClassification
import mymodel
import re
import json

import wave


def pcm2wav(pcm_file_list, wav_file, channels=1, bit_depth=16, sampling_rate=16000):
  # Check if the options are valid.
  if bit_depth % 8 != 0:
    raise ValueError("bit_depth " + str(bit_depth) + " must be a multiple of 8.")

  i = 0
  pcm_data = bytearray()
  # Read the .pcm file as a binary file and store the data to pcm_data
  for pcm_file in pcm_file_list:
    if (i == 0):

      with open(pcm_file, 'rb') as opened_pcm_file:
        pcm_data += opened_pcm_file.read()
    else:
      with open(pcm_file, 'rb') as opened_pcm_file:
        pcm_data += opened_pcm_file.read()

      #print(pcm_data)
      #print(type(pcm_data))

      obj2write = wave.open(wav_file, 'wb')
      obj2write.setnchannels(channels)
      obj2write.setsampwidth(bit_depth // 8)
      obj2write.setframerate(sampling_rate)
      obj2write.writeframes(pcm_data)
      obj2write.close()
    i += 1


labels = ["none", "joy", "annoy", "sad", "disgust", "surprise", "fear"]
none_words = ["안싫", "안 싫", "안무서", "안놀람", "안놀랐", "안행복", "안기뻐", "안빡", "안우울", "안짜증", "안깜짝", "안무섭"]
pass_words = ["안좋", "안 좋"]
senti_loss = [6.0, 4.0, 6.5, 6.5, 10.0, 9.0]

# file_num = 109


tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v3-discriminator")
# GPU 사용
device = torch.device("cuda")
model = mymodel.HwangariSentimentModel.from_pretrained("Kyuyoung11/haremotions-v2").to(device)

add_file_num = 0
total_text = ""
pcm_list = []
aa = 0

for a in range(1820, 1840, 1):
    joy_file = []
    surprise_file = []
    disgust_file = []
    fear_file = []

    file_num = a
    file_path = "./SDRW200000" + str(file_num) + ".json"
    print(file_path)

    with open(file_path, "rt", encoding='UTF8') as json_file:

        json_data = json.load(json_file)
        print(json_data)
        for json_docu in json_data["document"]:
            for json_string in json_docu["utterance"]:
                audio_id = json_string["id"]
                text = json_string["form"]



                if ("?" in text or "." in text):
                    pcm_list.append("./SDRW200000" + str(file_num) + "/" + audio_id + ".pcm")
                    total_text += text

                    enc = tokenizer.encode_plus(total_text)
                    inputs = tokenizer(
                        total_text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=256,
                        pad_to_max_length=True,
                        add_special_tokens=True
                    )

                    # print(tokenizer.tokenize(tokenizer.decode(enc["input_ids"])))

                    model.eval()

                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    output = model(input_ids.to(device), attention_mask.to(device))[0]
                    _, prediction = torch.max(output, 1)

                    label_loss_str = str(output).split(",")

                    label_loss = [float(x.strip().replace(']', '')) for x in label_loss_str[1:7]]

                    pre_result = int(re.findall("\d+", str(prediction))[0])
                    # 손실함수 값이 4.0이상인게 없으면 무감정(none)으로 분류
                    result = 0
                    if label_loss[pre_result - 1] >= senti_loss[pre_result - 1]:
                        result = pre_result

                    # 안이 들어간 말로 결과가 나왔을 경우 가장 큰 값을 무시함 or 아예 무감정으로 분류되도록 함
                    for i in none_words:
                        if i in total_text:
                            result = 0
                    for j in pass_words:
                        if j in total_text:
                            label_loss[pre_result - 1] = 0
                            result = label_loss.index(max(label_loss)) + 1


                    if (result == 1):
                        output_file = "joy/" + str(file_num) + "joy" + str(aa) + ".wav"
                        pcm2wav(pcm_list, output_file, 1, 16, 16000)
                        #joy_file.append(audio_id)
                        print("File name : " + audio_id)
                        print(f'Review text : {total_text}')

                        print(f'Sentiment : {labels[result]}')

                        print("\n<감정 별 손실 함수 값>")
                        for i in range(0, 6):
                            print(labels[i + 1], ":", label_loss[i])

                        print("----------------------------")

                    elif (result == 5):
                        output_file = "surprise/" + str(file_num) + "surprise" + str(aa) + ".wav"
                        pcm2wav(pcm_list, output_file, 1, 16, 16000)
                        #surprise_file.append(audio_id)
                        print("File name : " + audio_id)
                        print(f'Review text : {total_text}')

                        print(f'Sentiment : {labels[result]}')

                        print("\n<감정 별 손실 함수 값>")
                        for i in range(0, 6):
                            print(labels[i + 1], ":", label_loss[i])

                        print("----------------------------")

                    '''
                    elif (result == 4):
                        output_file = "disgust/" + str(file_num) + "disgust" + str(aa) + ".wav"
                        pcm2wav(pcm_list, output_file, 1, 16, 16000)
                        #disgust_file.append(audio_id)


                    elif (result == 6):
                        output_file = "fear/" + str(file_num) + "fear" + str(aa) + ".wav"
                        pcm2wav(pcm_list, output_file, 1, 16, 16000)
                        #fear_file.append(audio_id)
                    '''
                    aa+=1





                    pcm_list = []
                    total_text = ""
                    add_file_num = 0


                else:
                    pcm_list.append("./SDRW200000"+str(file_num)+"/"+audio_id+".pcm")

                    add_file_num += 1
                    total_text = total_text + " " + text

    aa = 0


'''
    print(file_num)
    file_list = []
    for i in range(0, len(joy_file)):
      
      file_name = "./SDRW200000"+str(file_num)+"/"+joy_file[i]+".pcm"
      file_list.append(file_name)
      output_file = "joy/"+str(file_num)+"joy"+str(i)+".wav"
      pcm2wav(file_list, output_file, 1, 16, 16000)
    

    for i in range(0, len(surprise_file)):
        file_name = "./SDRW200000" + str(file_num) + "/" + surprise_file[i] + ".pcm"
        output_file = "surprise/" + str(file_num) + "surprise" + str(i) + ".wav"
        pcm2wav(file_name, output_file, 1, 16, 16000)

    for i in range(0, len(disgust_file)):
        file_name = "./SDRW200000" + str(file_num) + "/" + disgust_file[i] + ".pcm"
        output_file = "disgust/" + str(file_num) + "disgust" + str(i) + ".wav"
        pcm2wav(file_name, output_file, 1, 16, 16000)

    for i in range(0, len(fear_file)):
        file_name = "./SDRW200000" + str(file_num) + "/" + fear_file[i] + ".pcm"
        output_file = "fear/" + str(file_num) + "fear" + str(i) + ".wav"
        pcm2wav(file_name, output_file, 1, 16, 16000)
'''

