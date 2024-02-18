import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wave
import librosa

sns.set(style="whitegrid")

def plot_wav(wav, band_name, color):
    # Visualize waveform using matplotlib and wave
    audio_file = wave.open(wav, 'rb')
    frame_rate = audio_file.getframerate()
    signal = np.frombuffer(audio_file.readframes(-1), dtype=np.int16)

    # Create a time array for the x axis
    time = np.linspace(0, len(signal) / frame_rate, num=len(signal))

    plt.figure(figsize=(20, 5))
    plt.plot(time, signal, color=color)
    plt.title(band_name,fontsize=20)
    plt.xlabel('Time (seconds)',fontsize=15)
    plt.ylabel('Amplitude',fontsize=15)
    plt.xlim(0, 10)
    plt.tight_layout()
    plt.show()