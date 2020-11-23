import logging
import warnings
import numpy as np
from scipy.fft import fft
from scipy.signal import fftconvolve, spectrogram
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Preprocessor_v1(BaseEstimator, TransformerMixin):
    def __init__(self,
                 sample_freq: float = 1 / 300,
                 cutoff: float = 50,
                 nperseg: int = 512,
                 spacing: int = 15):
        self.sample_freq = sample_freq
        self.cutoff = cutoff
        self.length = None
        self.freq = None
        self.times = None
        self.nperseg = nperseg
        self.spacing = spacing

    def transform(self, X, y=None):
        X_padded = np.nan_to_num(X)
        self.freqs, self.times, X_spec = spectrogram(X_padded,
                                                     fs=1 / self.sample_freq,
                                                     nperseg=self.nperseg,
                                                     noverlap=self.nperseg - self.spacing,
                                                     detrend=False,
                                                     scaling='spectrum')

        # for i in range(X_spec.shape[0]):
        #     f, ax = plt.subplots(figsize=(4.8, 2.4))
        #     ax.pcolormesh(self.times, self.freqs, 10 * np.log10(X_spec[i,]), cmap='viridis')
        #     ax.set_ylabel('Frequency [Hz]')
        #     ax.set_xlabel('Time [s]');
        #     plt.show()

        X_spec_cropped = X_spec[:, :np.argmax(self.freqs > 30)]

        return np.abs(X_spec_cropped)

    def fit(self, X, y=None):
        self.length = X.shape[1]
        return self


class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        assert X.dtype == np.float, f"Input type mismatch! Expected float but got {X.dtype}"
        # import pdb; pdb.set_trace()
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X, y=None):
        if np.any(np.isclose(self.std, 0)):
            logger.info("Some standard deviation values are 0")
        X = (X - self.mean) / np.where(np.isclose(self.std, 0), np.ones_like(self.std), self.std)
        return X

# class Crop(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         import pdb; pdb.set_trace()
#         return X

class Reducer(BaseEstimator, TransformerMixin):
    def __init__(self, stride: int = 2, kernel_size: int = 3):
        self.stride = stride
        self.kernel_size = kernel_size
        self.kernel = np.full(self.kernel_size, fill_value=1 / self.kernel_size)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # [(W−K+2P)/S]+1
        X_out = np.zeros((X.shape[0], int((X.shape[1] - self.kernel_size) / self.stride) + 1))
        for i in range(X.shape[0]):
            X_out[i,] = fftconvolve(X[i,], self.kernel, mode='valid')[::self.stride]
        return X_out


### Old functions

def preprocess(X, y, sample_freq: float = 1 / 300, cutoff: float = 30, samples=True):
    freq = np.fft.fftfreq(X.shape[1], d=sample_freq)
    cutoff_idx = np.argmax(freq > cutoff)

    X_prep = np.empty((X.shape[0], int((len(freq[:cutoff_idx]) - 3) / 2)))
    # Samples have different lengths, so we can not process them all at once
    for i in range(X.shape[0]):
        valid_X = X[i, ~np.isnan(X[i,])]
        N = len(valid_X)
        if cutoff_idx >= N / 2:
            warnings.warn(f"Cutoff frequency is higher then highest measurable frequency of {freq[N]}")
        # Filter out frequencies above the cutoff value
        # Note: If thWe are only interested in the positive frequencies
        # if np.maximum(freq) <= cutoff:
        #

        X_freq = np.abs(fft(valid_X)[1:cutoff_idx])
        X_prep[i,] = fftconvolve(X_freq, [1 / 3, 1 / 3, 1 / 3], mode='valid')[::2]
        #
        # X_prep[i,] = X_prep[:cutoff]

    # X_prep = Normalizer(norm='max').fit_transform(X_prep)
    X_prep = StandardScaler().fit_transform(X_prep)
    if samples:
        for i in range(50):
            plt.plot(X_prep[i,])
            plt.show()
    return X_prep, y
