import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
import librosa
import librosa.display
import warnings
warnings.filterwarnings('ignore')

def feat_mel_freq(y, hop_length, sr):
    """generate mfcc relevant features"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13, n_fft=441)
    mel_freq_features = np.round(
        np.array([np.mean(mfcc[0]), np.std(mfcc[0]), np.amin(mfcc[0]), np.amax(mfcc[0]), np.median(mfcc[0]),
                  np.mean(mfcc[1]), np.std(mfcc[1]), np.amin(mfcc[1]), np.amax(mfcc[1]), np.median(mfcc[1]),
                  np.mean(mfcc[2]), np.std(mfcc[2]), np.amin(mfcc[2]), np.amax(mfcc[2]), np.median(mfcc[2]),
                  np.mean(mfcc[3]), np.std(mfcc[3]), np.amin(mfcc[3]), np.amax(mfcc[3]), np.median(mfcc[3]),
                  np.mean(mfcc[4]), np.std(mfcc[4]), np.amin(mfcc[4]), np.amax(mfcc[4]), np.median(mfcc[4]),
                  np.mean(mfcc[5]), np.std(mfcc[5]), np.amin(mfcc[5]), np.amax(mfcc[5]), np.median(mfcc[5]),
                  np.mean(mfcc[6]), np.std(mfcc[6]), np.amin(mfcc[6]), np.amax(mfcc[6]), np.median(mfcc[6]),
                  np.mean(mfcc[7]), np.std(mfcc[7]), np.amin(mfcc[7]), np.amax(mfcc[7]), np.median(mfcc[7]),
                  np.mean(mfcc[8]), np.std(mfcc[8]), np.amin(mfcc[8]), np.amax(mfcc[8]), np.median(mfcc[8]),
                  np.mean(mfcc[9]), np.std(mfcc[9]), np.amin(mfcc[9]), np.amax(mfcc[9]), np.median(mfcc[9]),
                  np.mean(mfcc[10]), np.std(mfcc[10]), np.amin(mfcc[10]), np.amax(mfcc[10]), np.median(mfcc[10]),
                  np.mean(mfcc[11]), np.std(mfcc[11]), np.amin(mfcc[11]), np.amax(mfcc[11]), np.median(mfcc[11]),
                  np.mean(mfcc[12]), np.std(mfcc[12]), np.amin(mfcc[12]), np.amax(mfcc[12]), np.median(mfcc[12])]), 4)
    return mel_freq_features


def feat_f0(y):
    """generate f0 relevant features"""
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_features = np.round(
        np.array([np.amin(f0), np.amax(f0), np.mean(f0), np.std(f0), np.median(f0), kurtosis(f0), skew(f0)]), 4)
    return (f0_features)


def read_path(path):
    """read files as a list under path"""
    return os.listdir(path)

def read_audio(path, sr):
    """read audio, no transformation"""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y


def read_audio_male(path, sr):
    """read audio every 6s"""
    length = librosa.get_duration(filename=path)
    ys = []
    y, sr = librosa.load(path, sr=sr, mono=True)
    if length > 6:
        for i in range(int(length // 6)):
            m = y[i * sr:(i + 6) * sr]
            ys.append(m)
    else:
        ys.append(y)
    return ys


def read_audio_female(path, sr):
    """read audio every 6s with 3s stride"""
    length = librosa.get_duration(filename=path)
    ys = []
    y, sr = librosa.load(path, sr=sr, mono=True)
    step = 3
    if length > 6:
        for i in range(0, int(length), step):
            m = y[i * sr:(i + 6) * sr]
            ys.append(m)
    else:
        ys.append(y)
    return ys


def gen_feat(path, hop_length, sr):
    """generate features for training"""
    folders = read_path(path)
    data_fin = {"f0": [], "mfcc": [], "gender": []}
    for type in folders:
        current_path = path + "/" + type
        datanames = read_path(current_path)
        for dataname in datanames:
            if os.path.splitext(dataname)[-1] in ['.m4a', '.wav', '.mp3']:
                audio_path = current_path + "/" + dataname
                if type == "females":
                    female_ys = read_audio_female(audio_path, sr)
                    for y in female_ys:
                        f0 = feat_f0(y)
                        mfcc = feat_mel_freq(y, hop_length, sr)
                        data_fin['f0'].append(f0)
                        data_fin['mfcc'].append(mfcc)
                        data_fin['gender'].append(0)
                if type == "males":
                    male_ys = read_audio_male(audio_path, sr)
                    for y in male_ys:
                        f0 = feat_f0(y)
                        mfcc = feat_mel_freq(y, hop_length, sr)
                        data_fin['f0'].append(f0)
                        data_fin['mfcc'].append(mfcc)
                        data_fin['gender'].append(1)
    return data_fin


def gen_pre_feat(path, hop_length, sr):
    """generate features for prediction"""
    data_fin = {"f0": None, "mfcc": None, "gender": None}
    y = read_audio(path, sr)
    f0 = feat_f0(y)
    mfcc = feat_mel_freq(y, hop_length, sr)
    data_fin['f0'] = f0
    data_fin['mfcc'] = mfcc
    return data_fin


def gen_df(data):
    """transform features to dataframe for training"""
    d = pd.DataFrame(data)
    m1 = d['f0'].apply(pd.Series,
                       index=['f0_min', 'f0_max', 'f0_mean', 'f0_std', 'f0_median', 'f0_kurtosis', 'f0_skew'])
    mfcc_indice = [x for x in range(13)]
    stats = ['mean', 'std', 'min', 'max', 'median']
    cols = []
    for i in mfcc_indice:
        for s in stats:
            cols.append("mfcc_" + str(i) + "_" + s)
    m2 = d['mfcc'].apply(pd.Series, index=cols)
    m2['gender'] = d['gender']
    df = pd.concat([m1, m2], axis=1)
    print(f'train size: {df.shape}')
    return df

def gen_ins(data):
    """transform features to dataframe for prediction"""
    data = np.append(data['f0'], data['mfcc']).reshape(1, -1)
    df = pd.DataFrame(data)
    print(f'test size: {df.shape}')
    return df


def df2csv(df):
    """transform dataframe to csv"""
    df.to_csv('./df.csv')


def feat_engineering(path, flag, hop_length=220, sr=22050):
    """preprocessing for training set/prediction set"""
    if flag:
        data = gen_feat(path, hop_length, sr)
        df = gen_df(data)
        df2csv(df)
    else:
        data = gen_pre_feat(path, hop_length, sr)
        df = gen_ins(data)
    return df

