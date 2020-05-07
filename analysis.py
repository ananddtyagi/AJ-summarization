
import os
import json
import sys
import pickle
import numpy as np
from tqdm import tqdm
from tqdm.auto import trange
import pandas as pd
import matplotlib.pyplot as plt


def main():

    with open('./logs/percentile.txt', 'rb') as file:
        percentile = pickle.load(file)

    df = pd.Series(percentile)
    count = df.value_counts(bins=5) / len(df)

    count.plot()
    plt.show(block=True)


main()
