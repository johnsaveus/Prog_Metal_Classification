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

def plot_zcr(wavs, band_names):

    plt.figure(figsize=(10, 6))
    for wav_file, band in zip(wavs, band_names):
    # Load the audio file
        y, sr = librosa.load(wav_file, sr=None)
        
        # Calculate the zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Generate time axis for plotting
        t = np.linspace(0, len(y) / sr, num=len(zcr))
        
        # Plot the zero crossing rate
        plt.plot(t, zcr, label=band)

        # Adding labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Zero Crossing Rate')
    plt.title('Zero Crossing Rate of Four Different Labels')
    plt.legend()
    plt.xlim(0,10)

    # Show the plot
    plt.show()


def plot_mfcc(wav_files,labels):

    fig, axs = plt.subplots(nrows=len(wav_files), ncols=1, figsize=(10, 4 * len(wav_files)), sharex=True)

    for ax, wav_file, label in zip(axs, wav_files, labels):
        # Load the audio file
        y, sr = librosa.load(wav_file, sr=None)
        
        # Calculate MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Plotting the MFCCs
        img = librosa.display.specshow(mfccs, x_axis='time', ax=ax, sr=sr)
        ax.set_title(f'MFCCs for {label}')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

    # Set common labels
    plt.xlabel('Time (s)')
    plt.xlim(0,10)
    plt.ylabel('MFCC')
    plt.tight_layout()
    plt.show()


def plot_centroid(wav_files, labels):
    plt.figure(figsize=(10, 6))
    for wav_file, label in zip(wav_files, labels):
        y, sr = librosa.load(wav_file, sr=None)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        times = librosa.times_like(centroid)
        plt.plot(times, centroid[0], label=label)

    plt.xlabel('Time (s)')
    plt.ylabel('Spectral Centroid (Hz)')
    plt.title('Spectral Centroid Over Time')
    plt.xlim(0,10)
    plt.legend()
    plt.show()

def plot_bandwidth(wav_files, labels):
    plt.figure(figsize=(10, 6))
    for wav_file, label in zip(wav_files, labels):
        y, sr = librosa.load(wav_file, sr=None)

        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        times = librosa.times_like(bandwidth)
        plt.plot(times, bandwidth[0], label=label)

    plt.xlabel('Time (s)')
    plt.ylabel('Spectral Bandwidth (Hz)')
    plt.title('Spectral Bandwidth Over Time')
    plt.legend()
    plt.xlim(0,10)
    plt.show()

def plot_flux(wav_files, labels):
    plt.figure(figsize=(10, 6))
    
    for wav_file, label in zip(wav_files, labels):
        y, sr = librosa.load(wav_file, sr=None)
        # Compute the spectrogram magnitude and its first-order difference
        S = np.abs(librosa.stft(y))
        S_diff = np.abs(np.diff(S, axis=1))
        flux = np.sum(S_diff, axis=0)
        
        times = librosa.times_like(flux, sr=sr, hop_length=512)
        plt.plot(times, flux, label=label)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Spectral Flux')
    plt.title('Spectral Flux Over Time')
    plt.legend()
    plt.xlim(0,10)
    plt.show()

def plot_rolloff(wav_files, labels):
    plt.figure(figsize=(10, 6))
    
    for wav_file, label in zip(wav_files, labels):
        y, sr = librosa.load(wav_file, sr=None)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        times = librosa.times_like(rolloff)
        plt.plot(times, rolloff[0], label=label)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Spectral Rolloff (Hz)')
    plt.title('Spectral Rolloff Over Time')
    plt.legend()
    plt.xlim(0,10)
    plt.show()