import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.expand_frame_repr', False)

if __name__ == '__main__':
    TYPE = sys.argv[1]
    benchmark_csv = 'results/' + TYPE + '.csv'

    df = pd.read_csv(benchmark_csv, index_col='Dataset').transpose()
    print(df)

    for i in df.columns:
        plt.plot(df.index,df[i], label = i)
    plt.xlabel("Threshold")
    plt.ylabel("PA")
    plt.legend()
    plt.show()

    average = np.average(df,axis=1)
    y = average
    plt.plot(df.index,average)
    plt.xlabel("Threshold")
    plt.ylabel("Average PA")
    plt.show()

    # t = 0.75
    print(pd.DataFrame([df.index,average]))

