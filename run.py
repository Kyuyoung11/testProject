import re
import json

import wave
import classification_audiotext

#pcm 파일을 wav로 바꿈
# The parameters are prerequisite information. More specifically,
# channels, bit_depth, sampling_rate must be known to use this function.
def pcm2wav(pcm_file, wav_file, channels=1, bit_depth=16, sampling_rate=16000):
    # Check if the options are valid.
    if bit_depth % 8 != 0:
        raise ValueError("bit_depth " + str(bit_depth) + " must be a multiple of 8.")

    # Read the .pcm file as a binary file and store the data to pcm_data
    with open(pcm_file, 'rb') as opened_pcm_file:
        pcm_data = opened_pcm_file.read();

        obj2write = wave.open(wav_file, 'wb')
        obj2write.setnchannels(channels)
        obj2write.setsampwidth(bit_depth // 8)
        obj2write.setframerate(sampling_rate)
        obj2write.writeframes(pcm_data)
        obj2write.close()


classification = classification_audiotext.audioClassification()

for a in range(2030,2070,1):
  joy_file = []
  surprise_file = []
  disgust_file = []
  fear_file = []

  file_num = a
  file_path = "./SDRW200000"+str(file_num)+".json"
  with open(file_path,"rt", encoding='UTF8') as json_file:
    json_data = json.load(json_file)
    print(json_data)
    for json_docu in json_data["document"]:
      for json_string in json_docu["utterance"]:
        audio_id = json_string["id"]
        text = json_string["form"]
        if (len(text) <= 15) : continue

        file_name = "./SDRW200000" + str(file_num) + "/" + audio_id + ".pcm"
        output_file = "./pcm2wav.wav"
        pcm2wav(file_name, output_file, 1, 16, 16000)

        result = classification.classify(output_file, text)
        print(result)

