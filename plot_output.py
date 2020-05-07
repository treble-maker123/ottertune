from pdb import set_trace

import numpy as np
from matplotlib import pyplot as plt


def main():
    path = 'outputs/y_and_y_hat.csv'

    with open(path, 'r') as f:
        all_data = [t.strip().split(',') for t in f.readlines()]

    y = np.array([i[0] for i in all_data]).astype(float)
    y_hat = np.array([i[1] for i in all_data]).astype(float)

    line = np.arange(np.concatenate([y, y_hat]).max())

    plt.xlabel('Y')
    plt.ylabel('Y Hat')
    plt.scatter(y, y_hat)
    plt.plot(line, line)
    # plt.show()
    plt.savefig('outputs/y_and_y_hat.png')


if __name__ == '__main__':
    main()
