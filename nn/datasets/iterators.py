import os
import time
import numpy as np
import tensorflow as tf

class DataIterator:

    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

        self.num_examples = self.X['frames'].shape[0]
        self.epochs_completed = 0
        self.indices = np.arange(self.num_examples)
        self.reset_iteration()

    def reset_iteration(self):
        np.random.shuffle(self.indices)
        self.start_idx = 0

    def get_epoch(self):
        return self.epochs_completed

    def reset_epoch(self):
        self.reset_iteration()
        self.epochs_completed = 0

    def next_batch(self, batch_size, data_type="train", shuffle=True):#
        assert data_type in ["train", "val", "test"], \
            "data_type must be 'train', 'val', or 'test'."

        idx = self.indices[self.start_idx:self.start_idx + batch_size]

        batch_x = {}
        for key in self.X.keys():
            batch_x[key] = self.X[key][idx]

        batch_y = self.Y[idx] if self.Y is not None else self.Y
        self.start_idx += batch_size

        if self.start_idx + batch_size > self.num_examples:
            self.reset_iteration()
            self.epochs_completed += 1

        return (batch_x, batch_y)
    
    def sample_random_batch(self, batch_size):
        start_idx = np.random.randint(0, self.num_examples - batch_size)

        batch_x = {}
        for key in self.X.keys():
            batch_x[key] = self.X[key][self.start_idx:self.start_idx + batch_size]

        batch_y = self.Y[self.start_idx:self.start_idx + batch_size] if self.Y is not None else self.Y
        
        return (batch_x, batch_y)


def get_iterators(file, conv=False, datapoints=0):
    data = dict(np.load(file, allow_pickle=True))
    data['train_x'] = data['train_x'].item()
    data['valid_x'] = data['valid_x'].item()
    data['test_x'] = data['test_x'].item()
    if conv:
        img_shape = data["train_x"]['frames'][0,0].shape
    else:
        img_shape = data["train_x"]['frames'][0,0].flatten().shape
    data['train_x']['frames'] = data["train_x"]['frames'].reshape(data["train_x"]['frames'].shape[:2]+img_shape)/255
    data['valid_x']['frames'] = data["valid_x"]['frames'].reshape(data["valid_x"]['frames'].shape[:2]+img_shape)/255
    data['test_x']['frames'] = data["test_x"]['frames'].reshape(data["test_x"]['frames'].shape[:2]+img_shape)/255
    train_it = DataIterator(X=data['train_x'])
    valid_it = DataIterator(X=data["valid_x"])
    test_it = DataIterator(X=data["test_x"])
    return train_it, valid_it, test_it
