import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("./freqs.csv",
                sep = ',',
                encoding = 'utf-8')

def ShowBarPlot():
    global df
    names = df.columns[:16].to_list()
    freqs = df.iloc[0, :16].to_list()

    fig, ax = plt.subplots()
    ax.bar(names, freqs)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation = 45, ha = "right")

    ax.set_ylabel('frenquency')
    ax.set_xlabel('4-grams')
    ax.set_title('Title')

    %matplotlib inline
    plt.show()

    return

ShowBarPlot()