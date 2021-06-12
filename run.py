import re
import json

import wave
import classification_audiotext

def pcm2wav(pcm_file_list, wav_file, channels=1, bit_depth=16, sampling_rate=16000):
  # Check if the options are valid.
  if bit_depth % 8 != 0:
    raise ValueError("bit_depth " + str(bit_depth) + " must be a multiple of 8.")

  i = 0
  pcm_data = bytearray()
  # Read the .pcm file as a binary file and store the data to pcm_data
  for pcm_file in pcm_file_list:
    with open(pcm_file, 'rb') as opened_pcm_file:
      pcm_data += opened_pcm_file.read()

  obj2write = wave.open(wav_file, 'wb')
  obj2write.setnchannels(channels)
  obj2write.setsampwidth(bit_depth // 8)
  obj2write.setframerate(sampling_rate)
  obj2write.writeframes(pcm_data)
  obj2write.close()


classification = classification_audiotext.audioClassification()

add_file_num = 0
total_text = ""
pcm_list = []
aa = 0

for a in range(1180, 1200, 1):
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
                    output_file = './audio.wav'
                    pcm2wav(pcm_list, output_file, 1, 16, 16000)
                    total_text += text

                    result = classification.classify(output_file, total_text)


                    if (result == 1):
                        output_file = "joy/" + str(file_num) + "joy" + str(aa) + ".wav"
                        pcm2wav(pcm_list, output_file, 1, 16, 16000)


                        print(f'Review text : {total_text}')

                        print(f'Sentiment : {classification.labels[result]}')



                    elif (result == 5):
                        output_file = "surprise/" + str(file_num) + "surprise" + str(aa) + ".wav"
                        pcm2wav(pcm_list, output_file, 1, 16, 16000)
                        #surprise_file.append(audio_id)
                        print(f'Review text : {total_text}')

                        print(f'Sentiment : {classification.labels[result]}')



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

