import wave

#pcm 파일을 wav로 바꿈
# The parameters are prerequisite information. More specifically,
# channels, bit_depth, sampling_rate must be known to use this function.
def pcm2wav(pcm_file_list, wav_file, channels=1, bit_depth=16, sampling_rate=16000):
    # Check if the options are valid.
    if bit_depth % 8 != 0:
        raise ValueError("bit_depth " + str(bit_depth) + " must be a multiple of 8.")

    i = 0
    pcm_data = bytearray()
    # Read the .pcm file as a binary file and store the data to pcm_data
    for pcm_file in pcm_file_list:
        if (i == 0 ):

            with open(pcm_file, 'rb') as opened_pcm_file:
                pcm_data += opened_pcm_file.read()
        else :
            with open(pcm_file, 'rb') as opened_pcm_file:
                pcm_data += opened_pcm_file.read()

            print(pcm_data)
            print(type(pcm_data))

            obj2write = wave.open(wav_file, 'wb')
            obj2write.setnchannels(channels)
            obj2write.setsampwidth(bit_depth // 8)
            obj2write.setframerate(sampling_rate)
            obj2write.writeframes(pcm_data)
            obj2write.close()
        i+=1


pcm_file_list = []
pcm_file_list.append('./SDRW2000002030/SDRW2000002030.1.1.1.pcm')
pcm_file_list.append('./SDRW2000002030/SDRW2000002030.1.1.2.pcm')
pcm_file_list.append('./SDRW2000002030/SDRW2000002030.1.1.3.pcm')
pcm2wav(pcm_file_list, 'audio1.wav', 1, 16, 16000)