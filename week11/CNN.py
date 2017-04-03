from __future__ import print_function
import tensorflow as tf
import numpy as np
from cifar import load_cifar_file

images, labels = load_cifar_file('./input/data_batch_1')
images = 2*images.astype(np.float)/255. - 1
N = labels.shape[0]

#One-hot
labels_tmp = np.zeros((N, 10))
labels_tmp[np.arange(N), labels] = 1
labels = labels_tmp