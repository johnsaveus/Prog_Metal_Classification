import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wave
import librosa

sns.set(style="whitegrid")

def plot_wav(wav, band_name, color):
    # Load a WAV file (you can replace 'your_audio_file.wav' with your actual file)
    audio_file = wave.open(wav, 'rb')
    frame_rate = audio_file.getframerate()
    signal = np.frombuffer(audio_file.readframes(-1), dtype=np.int16)

    # Create a time array
    time = np.linspace(0, len(signal) / frame_rate, num=len(signal))

    # Plot using Matplotlib and customize with Seaborn
    plt.figure(figsize=(20, 5))
    
    sns.set(style="whitegrid")
    plt.plot(time, signal, color=color)
    plt.title(band_name,fontsize=20)
    plt.xlabel('Time (seconds)',fontsize=15)
    plt.ylabel('Amplitude',fontsize=15)
    plt.xlim(0, 10)
    plt.tight_layout()
    plt.show()