import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    param = 'length'
    default_value = 10.0

    lengths = []
    directory = f'./out_{param}'
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), 'r') as f:
            lines = f.read().split('\n')
            ls = [float(line) for line in lines if line != '']
            lengths.append(ls)

    for i, length in enumerate(lengths):
        plt.plot(length, label=f'run {i}')
    plt.xlabel('Iteration')
    plt.ylabel(param)
    plt.legend()
    plt.title(f'{param} (default={default_value}) over iterations')
    plt.show()


