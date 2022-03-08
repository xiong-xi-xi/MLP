import numpy as np
import gzip
import random

train_size = 60000
test_size = 10000
pic_size = 28 * 28

file_dict = {
    'train_img': './dataset/train-images-idx3-ubyte.gz',
    'train_label': './dataset/train-labels-idx1-ubyte.gz',
    'test_img': './dataset/t10k-images-idx3-ubyte.gz',
    'test_label': './dataset/t10k-labels-idx1-ubyte.gz'
}

def _load_img_label(file_name, offset):
    file_path = file_name
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=offset)

    return data

def _load_img(img_file):
    img_data = _load_img_label(file_dict[img_file], 16)
    img_data = img_data.astype(np.float32) / 255.0
    img_data = img_data.reshape(-1, 1, pic_size)

    return img_data

def _load_label(label_file):
    label_data = _load_img_label(file_dict[label_file], 8)
    label_data = _one_hot_trans(label_data, 10)

    return label_data

def _one_hot_trans(label, classes):
    Label = np.zeros((label.size, classes))
    for idx in range(label.size):
        Label[idx][label[idx]] = 1

    return Label

def load_dataset():
    dataset = {}
    for key in file_dict.keys():
        if key in ['train_img', 'test_img']:
            dataset[key] = _load_img(key)

        if key in ['train_label', 'test_label']:
            dataset[key] = _load_label(key)

    ### shuffle step
    index = [i for i in range(train_size)]
    random.shuffle(index)
    train_img = dataset['train_img'][index]
    train_label = dataset['train_label'][index]

    test_img = dataset['test_img']
    test_label = dataset['test_label']

    return (train_img, train_label), (test_img, test_label)
