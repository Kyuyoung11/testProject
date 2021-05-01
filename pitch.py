import librosa
import scipy.signal as signal
import numpy as np

audio_sample, sampling_rate = librosa.load("sine.wav", sr = None)

S = np.abs(librosa.stft(audio_sample, n_fft=1024, hop_length=512, win_length = 1024, window=signal.hann))
pitches, magnitudes = librosa.piptrack(S=S, sr=sampling_rate)

shape = np.shape(pitches)
nb_samples = shape[0]
nb_windows = shape[1]

for i in range(0, nb_windows):
    index = magnitudes[:,i].argmax()
    pitch = pitches[index,i]
    print(pitch)

# FFT 결과를 plot
import matplotlib.pyplot as plt
import librosa.display

#normalize_function
min_level_db = -100
def _normalize(S):
    return np.clip((S-min_level_db)/(-min_level_db), 0, 1)

mag_db = librosa.amplitude_to_db(S)
mag_n = _normalize(mag_db)
plt.subplot(311)
librosa.display.specshow(mag_n, y_axis='linear', x_axis='time', sr=sampling_rate)
plt.title('spectrogram')

t = np.linspace(0, 24000, mag_db.shape[0])
plt.subplot(313)
plt.plot(t, mag_db[:, 100].T)
plt.title('magnitude (dB)')
plt.show()