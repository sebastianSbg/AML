import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# print(df_train.shape)
# df_train.sample(100).to_csv("X_train_small.csv")
from preprocessing import Preprocessor_v1
if __name__ == "__main__":
    df_train = pd.read_csv("X_train_small.csv")
    df_train = df_train.set_index("id")

    prep = Preprocessor_v1(max_length=df_train.shape[1])
    X = prep.fit_transform(df_train.to_numpy())
    plt.plot(X[1,])
    plt.show()

    example = df_train.iloc[3,1:].to_numpy()

    plt.plot(np.arange(0, (len(example))/300, 1/300), example)
    plt.show()

    valid_example = example[~np.isnan(example)]
    T = 1.0 / 300.0
    N = len(valid_example)
    x = np.linspace(0.0, N*T, N)
    freq = np.fft.fftfreq(N, d=T)
    firstNegInd = np.argmax(freq < 0)
    y = fft(valid_example)/N
    plt.plot(freq[:firstNegInd],np.abs(y[:firstNegInd]))
    plt.show()
    print('end')
