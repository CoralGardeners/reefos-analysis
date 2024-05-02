# train and predict models to cluster mel spectrogram slices

from dataclasses import dataclass
import pandas as pd
import numpy as np

# imports needed for cluster model
import joblib
import librosa as lr
import umap
from hdbscan import HDBSCAN, membership_vector, all_points_membership_vectors

# influx-reltaed imports
from influxdb_client import Point
import reefos_analysis.dbutils.influx_util as iu


# spectrogram parameter dataclasses
@dataclass
class SpecParams:
    n_mels: int = 128
    ts_slice: float = 0.1
    overlap: float = 0.75
    fmax: int = 800


# clustering params
@dataclass
class ClusParams:
    S_threshold: int = 0
    n_components: int = 3
    min_clus_size: int = 6
    min_samples: int = 3
    cluster_selection_epsilon: float = 1
    distance: str = 'euclidean'


# compute mel spectrogram
def mel_spectrogram(ts, fs, tmin=None, tmax=None, fmax=5000, overlap=0.5, wd=0.1, n_mels=64, dBref=1.0):
    if tmin is not None and tmax is not None:
        ts = ts[int(tmin * fs):int(tmax * fs)]
    fmin = 1 / wd
    if fmax is None or fmax > (fs / 2):
        fmax = (fs / 2)
    nperseg = int(wd * fs)
    hop = int((1 - overlap) * nperseg)
    S = lr.feature.melspectrogram(y=ts, sr=fs, n_fft=nperseg, fmin=fmin, fmax=fmax,
                                  hop_length=hop, n_mels=n_mels)
    mel_freq = lr.core.mel_frequencies(n_mels, fmin=fmin, fmax=fmax)
    S_dB = lr.power_to_db(S, ref=dBref)
    x = np.arange(0, len(ts) + 1, hop) / 48000
    return S_dB, x, mel_freq


def filter_data(S_dB, x, fileids=None, xvals=None, cp=None):
    if cp is None:
        cp = ClusParams()
    # only keep timestamps where max value is above a threshold level
    mask = S_dB.max(axis=0) > cp.S_threshold
    print(f"Number of slices with peak amplitude > {cp.S_threshold}: {mask.sum()}")
    times = x[mask]
    fileids = None if fileids is None else fileids[mask]
    xvals = None if xvals is None else xvals[mask]
    filtered_S = S_dB[:, mask]
    return filtered_S, times, fileids, xvals


# on the mel spectrogram data, filter by amplitude and cluster by spectral similarity
def train_models(S_dB, x, xvals=None, fileids=None, plot_dist=True, cp=None):
    if cp is None:
        cp = ClusParams()
    print(f"Training umap with {cp.distance}")
    X_train = S_dB.T
    # reduce dimensionality using umap
    # use parameters recommended in umap docs for cluster analysis
    n_neighbors = np.clip(X_train.shape[0] // 200, 8, 50)
    umap_model = umap.UMAP(n_components=cp.n_components, min_dist=0,
                           n_neighbors=n_neighbors, metric=cp.distance)
    umap_model.fit(X_train)
    umap_layout = umap_model.embedding_
    # train hdbscan model to cluster the umap embedding
    hdb_model = HDBSCAN(min_cluster_size=cp.min_clus_size, min_samples=cp.min_samples,
                        cluster_selection_epsilon=cp.cluster_selection_epsilon, prediction_data=True)
    hdb_model.fit(umap_layout)
    return umap_model, hdb_model


def run_models(umap_model, hdb_model, S_dB, times, fileids=None, xvals=None):
    # run umap and hdbscan on spectorgram data to assign each spectrogram slice to a group
    X_test = S_dB.T
    max_vals = S_dB.max(axis=0)
    umap_layout = umap_model.transform(X_test)

    soft_clusters = membership_vector(hdb_model, umap_layout)
    labels = [np.argmax(x) for x in soft_clusters]
    strengths = [np.max(x) for x in soft_clusters]

    # put umap results into a dataframe
    udf = pd.DataFrame({'max': max_vals,
                        'time': times,
                        'x': umap_layout[:, 0], 'y': umap_layout[:, 1],
                        'label': labels,
                        'label_prob': strengths
                        })
    if fileids is not None:
        udf['fileid'] = fileids
    if xvals is not None:
        udf['filetime'] = xvals
    # re-order the histogram slices by cluster label and time
    udf.sort_values(['label', 'time'], inplace=True)
    grouped_S = X_test[udf.index.values].T
    return udf, grouped_S


# make a dataset for grouping/clustering analysis from an in-memory time series
def make_timeseries_dataset(ts, fs, spec_params):
    audio_data = {}
    audio_data[0] = (ts, fs)
    final_S, final_x, y = mel_spectrogram(ts, fs, overlap=spec_params.overlap,
                                          fmax=spec_params.fmax, n_mels=spec_params.n_mels)
    final_t = final_x.copy()
    final_fileid = np.full(final_x.shape, 0)
    return final_x, final_t, final_S, final_fileid, y, fs, audio_data


# compute mean spectrogram mel amplitudes for each group
# cluster each group based on those means, using umap and hdbscan
def cluster_audio_groups(all_udf, grouped_S, cl_df=None):
    S_chunks = {}
    df_chunks = {}
    S_means = []
    all_udf = all_udf.reset_index()
    for val, df in all_udf.groupby('label'):
        idx1 = df.index.min()
        idx2 = df.index.max()
        df_chunks[val] = df
        S_grp = grouped_S[:, idx1:(idx2 + 1)]
        S_chunks[val] = S_grp
        S_means.append(S_grp.mean(axis=1))

    S_means = np.stack(S_means)

    if cl_df is None:
        # do a 2-D umap embedding of the cluster centers to get the best ordering of the clusters
        umap_2D = umap.UMAP(metric='euclidean', n_components=2, min_dist=0, n_neighbors=10)
        umap_2D_layout = umap_2D.fit_transform(S_means)
        # use hdbscan to label clusters
        hdb2D = HDBSCAN(min_cluster_size=3, min_samples=1, cluster_selection_epsilon=0.4, prediction_data=True)
        hdb2D.fit(umap_2D_layout)
        soft_clusters = all_points_membership_vectors(hdb2D)
        labels = [np.argmax(x) for x in soft_clusters]
        cl_df = pd.DataFrame({'x': umap_2D_layout[:, 0], 'y': umap_2D_layout[:, 1], 'cluster': labels})
        cl_df = cl_df.reset_index().rename({'index': 'label'})
        cl_df.sort_values(['cluster', 'x'], inplace=True)

    # re-order the spectrogram
    order = cl_df.index.values

    ordered_Ss = []
    ordered_dfs = []
    for idx in order:
        if idx in df_chunks:
            df = df_chunks[idx]
            df['label'] = idx
            df['cluster'] = cl_df.loc[idx, 'cluster']
            ordered_Ss.append(S_chunks[idx])
            ordered_dfs.append(df)

    ordered_S = np.concatenate(ordered_Ss, axis=1)
    ordered_df = pd.concat(ordered_dfs)
    return ordered_S, ordered_df, cl_df


########
# The prediction is a four step process applied to the audio time series:
# 1) Comp[ute mel spectrogram and isolate high amplitude events
# 2] Use the pre-trained umap model to compute the umap embedding of high amplitude mel spectrogram slices
# 3) Use the pretrained hbdscan model to assign hdbscan group labels to the umap embedding coordinates
# 4) Use a saved mapping (from an hdbscan model) to assign cluster labels to the group labels
# This is split into two functions so that step (1) can be in-memory (implemented here) or from a set of files
def predict_groups_clusters_from_spectrograms(models, x, times, mS, fileids, y):
    # data is either a list of filenames or an in-memory time series and sample rate
    cl_params = models['cl_params']
    umap_model = models['umap_model']
    hdb_model = models['hdb_model']
    cl_df = models['cl_df']

    # filter the data to capture 'loud' events
    filtered_S, times, fileids, xvals = filter_data(mS, times, fileids=fileids, xvals=x, cp=cl_params)
    # (2) compute embedding coordinates and (3) assign group labels to the data
    all_udf, grouped_S = run_models(umap_model, hdb_model, filtered_S, times,
                                    fileids=fileids, xvals=xvals)
    # (4) assign clusters to the group-labeled data
    ordered_S, ordered_df, cl_df = cluster_audio_groups(all_udf, grouped_S, cl_df)
    return all_udf, grouped_S, ordered_df, ordered_S, y, cl_df


def predict_groups_clusters_from_timeseries(models, ts, fs):
    spec_params = models['spec_params']
    all_x, all_t, all_S, all_fileid, y, fs, audio_data = make_timeseries_dataset(ts, fs, spec_params)
    all_udf, grouped_S, ordered_df, ordered_S, y, cl_df = \
        predict_groups_clusters_from_spectrograms(models, all_x, all_t, all_S, all_fileid, y)
    return all_udf, grouped_S, ordered_df, ordered_S, y, cl_df, audio_data, fs


def get_timeseries_clusters(ts, fs, models_fname):
    # get cluster counds in the chunk of audio
    drop_cols = ['fileid', 'filetime', 'index', 'x', 'y']
    models = joblib.load(models_fname)
    res = predict_groups_clusters_from_timeseries(models, ts, fs)
    rdf = res[2].drop(columns=drop_cols)
    # get cluster count dict
    vals = rdf.cluster.value_counts().to_dict()
    # make return dict with zero values for clusters not in the results
    all_clus = res[5].cluster.unique()
    return {f"Cluster_{clus}": vals[clus] if clus in vals else 0 for clus in all_clus}


def update_influx_bioacoustics_clusters(clusters, timestamp, env, version="0.1"):
    write_api = iu.setup_influx(env.influxdb_url, env.influxdb_token, env.influxdb_org)
    point = Point("bioacoustics_clusters").tag("version", version)
    for idx, val in clusters.items():
        point = point.field(idx, val)
    write_api.write(bucket=env.bucket_name, record=point)
