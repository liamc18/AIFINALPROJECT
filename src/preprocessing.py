import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess(audio_path, sr=22050, n_mels=128):
    y, sr = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return y, S_dB

def time_stretch_audio(y, rate=1.25):
    return librosa.effects.time_stretch(y, rate)

def pitch_shift_audio(y, sr, n_steps=4):
    return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)

def save_spectrogram(spectrogram, output_path, sr=22050):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency Spectrogram')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    audio_dir = os.path.join("..", "data", "audio_files")
    spec_output_dir = os.path.join("..", "data", "spectrograms")
    os.makedirs(spec_output_dir, exist_ok=True)
    
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            audio_path = os.path.join(audio_dir, filename)
            y, spec = load_and_preprocess(audio_path)
            output_file = os.path.join(spec_output_dir, filename.split('.')[0] + "_spec.png")
            save_spectrogram(spec, output_file)
            
            y_stretched = time_stretch_audio(y, rate=1.2)
            _, spec_stretched = load_and_preprocess(audio_path, sr=22050)
            output_file_stretched = os.path.join(spec_output_dir, filename.split('.')[0] + "_stretched_spec.png")
            save_spectrogram(spec_stretched, output_file_stretched)
            
            y_shifted = pitch_shift_audio(y, sr=22050, n_steps=2)
            _, spec_shifted = load_and_preprocess(audio_path, sr=22050)
            output_file_shifted = os.path.join(spec_output_dir, filename.split('.')[0] + "_shifted_spec.png")
            save_spectrogram(spec_shifted, output_file_shifted)
