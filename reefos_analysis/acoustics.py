import numpy as np
from scipy.signal import spectrogram, find_peaks
from scipy.stats import entropy

import io
import soundfile as sf
from influxdb_client import Point

from reefos_analysis.dbutils.influx_util import setup_influx

# %%
# acoustic index parameters
default_ai_params = {
    'ADI': {'threshold': 1e-5},
    'BI': {'lofreq': 50, 'hifreq': 2000, 'threshold': 1e-7},
    'NDSI': {'lofreq': 50, 'midfreq': 500, 'hifreq': 5000},
}


def calculate_acoustic_indices(audio_data, sample_rate, max_freq=None, ai_params=None,
                               window_len=2048, frac_overlap=0.5):
    ai_parameters = ai_params or default_ai_params
    # Generate the spectrogram
    noverlap = int(window_len * frac_overlap)
    frequencies, times, Sxx = spectrogram(audio_data, fs=sample_rate,
                                          window="hann", noverlap=noverlap, nperseg=window_len)
    if max_freq is not None and max_freq < sample_rate / 2:
        mask = frequencies < max_freq
        frequencies = frequencies[mask]
        Sxx = Sxx[mask]

    indices = {}

    # Temporal Acoustic Entropy (Ht)
    Ht = entropy(np.sum(Sxx, axis=0))
    indices['Ht'] = round(Ht, 2)

    # Spectral Acoustic Entropy (Hf)
    Hf = entropy(np.sum(Sxx, axis=1))
    indices['Hf'] = round(Hf, 2)

    # Acoustic Diversity Index (ADI)
    params = ai_parameters['ADI']
    num_bins = 10
    bin_size = int(Sxx.shape[0] / num_bins)
    adi_values = []
    for i in range(num_bins):
        bin_data = Sxx[i*bin_size:(i+1)*bin_size, :]
        proportion = np.sum(bin_data > params['threshold']) / bin_data.size
        adi_values.append(proportion)
    ADI = entropy(adi_values) if sum(adi_values) > 0 else 0
    indices['ADI'] = round(ADI, 2)

    # Acoustic Complexity Index (ACI)
    diff_matrix = np.abs(np.diff(Sxx, axis=1))
    ACI = np.sum(diff_matrix) / np.sum(Sxx)
    indices['ACI'] = round(ACI, 2)

    # Bioacoustics Index (BI)
    params = ai_parameters['BI']
    freq_indices = (frequencies >= params['lofreq']) & (frequencies <= params['hifreq'])
    BI = np.sum(Sxx[freq_indices, :] > params['threshold']) / Sxx[freq_indices, :].size
    indices['BI'] = round(BI, 2)

    # Number of detected frequency peaks (NBPEAKS)
    mean_spectrum = np.mean(Sxx, axis=1)
    peaks, _ = find_peaks(mean_spectrum)
    NBPEAKS = len(peaks)
    indices['NBPEAKS'] = NBPEAKS

    # Normalized Difference Soundscape Index (NDSI)
    params = ai_parameters['NDSI']
    anthrophony = np.sum(Sxx[(frequencies > params['lofreq']) & (frequencies < params['midfreq']), :])
    biophony = np.sum(Sxx[(frequencies > params['midfreq']) & (frequencies < params['hifreq']), :])
    NDSI = (biophony - anthrophony) / (biophony + anthrophony)
    indices['NDSI'] = round(NDSI, 2)

    # Signal-to-Noise Ratio across the spectral domain (SNRf)
    signal_f = np.max(Sxx, axis=1)
    noise_f = np.min(Sxx, axis=1)
    SNRf = 10 * np.log10(np.sum(signal_f) / np.sum(noise_f))
    indices['SNRf'] = round(SNRf, 2)

    # Signal-to-Noise Ratio across the temporal domain (SNRt)
    signal_t = np.max(Sxx, axis=0)
    noise_t = np.min(Sxx, axis=0)
    SNRt = 10 * np.log10(np.sum(signal_t) / np.sum(noise_t))
    indices['SNRt'] = round(SNRt, 2)

    return indices


def update_influx_bioacoustics(indices, timestamp, env, version="0.1"):
    write_api = setup_influx(env.influxdb_url, env.influxdb_token, env.influxdb_org)
    point = Point("bioacoustics").tag("version", version)
    for idx, val in indices.items():
        point = point.field(idx, val)
    write_api.write(bucket=env.bucket_name, record=point)


# %%
if __name__ == '__main__':
    # datapath = "Audio/Data/reefos-02_audio_10_17_2023_10_37_31.wav"
    # datapath = "Audio/Data/reefos-01_audio_11_06_2023_12_55_47.wav"
    # datapath = "Audio/Data/reefos-01_audio_11_05_2023_08_41_51.wav"
    datapath = "Audio/Data/reefos-01_audio_11_06_2023_15_08_08.wav"
    with open(datapath, 'rb') as audio_file:
        aa, sample_rate = sf.read(io.BytesIO(audio_file.read()))
        indices = calculate_acoustic_indices(aa, sample_rate)
