import os

import numpy as np
import pandas as pd
from sklearn.datasets import *
from sklearn.datasets.base import Bunch

mnist_rot_file_base = 'mnist_all_rotation_normalized_float_'# 'mnist_all_background_images_rotation_normalized_'
mnist_back_file_base = 'mnist_background_images_'
#mnist_back_file_base = 'mnist_background_random_'


def load(name):
    LOAD = {
        'iris': load_iris(),
        'digits': load_digits(),
        'census_csv': load_census_50k(),
        'wine': load_wine(),
        'breast_cancer': load_breast_cancer(),
        'mnist': load_mnist(),
        'mnist_test': load_mnist_test(),
        'mnist-rot': load_mnist_rot(),
        'mnist-rot-test': load_mnist_rot_test(),
    }
    return LOAD[name]

def load_census_50k(return_X_y=False):
    filename = 'census_data.csv'
    path = os.path.join('../SVM/census',filename)
    data = pd.read_csv(path)
    data = data[data.occupation != '?']

    data['workclass_num'] = data.workclass.map({
        'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3,
        'Self-emp-inc':4, 'Local-gov':5, 'Without-pay':6})
    data['over50K'] = np.where(data.income == '<=50K', 0, 1)
    data['marital_num'] = data['marital.status'].map({
        'Widowed':0, 'Divorced':1, 'Separated':2, 'Never-married':3,
        'Married-civ-spouse':4, 'Married-AF-spouse':4, 'Married-spouse-absent':5})
    data['race_num'] = data.race.map({
        'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4})
    data['sex_num'] = np.where(data.sex == 'Female', 0, 1)
    data['rel_num'] = data.relationship.map(
        {'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 'Husband':1, 'Wife':1})
    data.head()

    feature_names = ['workclass_num', 'education.num', 'marital_num', 'race_num', 'sex_num', 'rel_num', 'capital.gain', 'capital.loss']

    X = np.array(data[feature_names])
    y = np.array(data.over50K)

    if return_X_y:
        return X, y

    return Bunch(data=X, target=y,
                 target_names=['income <=50k', 'income >50k'],
                 DESCR=None,
                 feature_names=feature_names,
                 filename=path)

def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

def bunchFile(csv_file):
    file = open(csv_file)
    data_train = pd.read_csv(file)

    y= np.array(data_train.iloc[:, 0])
    X= np.array(data_train.iloc[:, 1:])

    return Bunch(data=X, target = y, DESCR = None, filename = csv_file)

def load_mnist_(imgs, labls, out, n):
    imgs = os.path.join('./data', imgs)
    labls = os.path.join('./data', labls)
    out = os.path.join('./data', out)
    if not os.path.isfile(out):
        convert(imgs, labls, out, n)
    return bunchFile(out)

def load_mnist_test(return_X_y=False):
    return load_mnist_("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",
        "mnist_test.csv", 10000)

def load_mnist(return_X_y=False):
    return load_mnist_("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
        "mnist_train.csv", 60000)

def mnist_rot_to_csv(csv_name, base, t):
    file = os.path.join('./data', base+t+".amat")

    df = pd.read_csv(file, sep='\s+', header=None)
    df.to_csv(csv_name, header=None)

def load_mnist_rot_test():
    csv_file = "./data/mnist-rot-test.csv"
    if not os.path.isfile(csv_file):
        mnist_rot_to_csv(csv_file, mnist_rot_file_base,'test')
    file = open(csv_file)
    data_train = pd.read_csv(file)

    y= np.array(data_train.iloc[:100, 785])
    X= np.array(data_train.iloc[:100, :784])

    return Bunch(data=X, target = y, DESCR = None, filename = csv_file)

def load_mnist_rot():
    csv_file = "./data/mnist-rot.csv"
    if not os.path.isfile(csv_file):
        mnist_rot_to_csv(csv_file, mnist_rot_file_base,'train_valid')
    file = open(csv_file)
    data_train = pd.read_csv(file)

    y= np.array(data_train.iloc[:100, 785])
    X= np.array(data_train.iloc[:100, :784])

    return Bunch(data=X, target = y, DESCR = None, filename = csv_file)

def load_mnist_back_test():
    csv_file = "./data/mnist-back-test.csv"
    if not os.path.isfile(csv_file):
        mnist_rot_to_csv(csv_file, mnist_back_file_base, 'test')
    file = open(csv_file)
    data_train = pd.read_csv(file)

    y= np.array(data_train.iloc[:100, 785])
    X= np.array(data_train.iloc[:100, :784])

    return Bunch(data=X, target = y, DESCR = None, filename = csv_file)

def load_mnist_back():
    csv_file = "./data/mnist-back.csv"
    if not os.path.isfile(csv_file):
        mnist_rot_to_csv(csv_file, mnist_back_file_base, 'train')
    file = open(csv_file)
    data_train = pd.read_csv(file)

    y= np.array(data_train.iloc[:100, 785])
    X= np.array(data_train.iloc[:100, :784])

    return Bunch(data=X, target = y, DESCR = None, filename = csv_file)

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
