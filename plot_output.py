from pdb import set_trace

import numpy as np
from matplotlib import pyplot as plt


def main():
    path = 'outputs/y_and_y_hat.csv'

    with open(path, 'r') as f:
        y, y_hat = f.readlines()

    y = y.replace('\n', '')
    y_hat = y_hat.replace('\n', '')

    y = np.array(y.split(',')).astype(float)
    y_hat = np.array(y_hat.split(',')).astype(float)

    line = np.arange(np.concatenate([y, y_hat]).max())

    plt.xlabel('Y')
    plt.ylabel('Y Hat')
    plt.scatter(y, y_hat)
    plt.plot(line, line)
    plt.show()


if __name__ == '__main__':
    main()
