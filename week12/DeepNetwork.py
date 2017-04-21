from __future__ import print_function
import tensorflow as tf
import numpy as np
from cifar import load_cifar_file
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import vgg

def get_data(datafile='../input/data_batch_1'):
  images, labels = load_cifar_file(datafile)
  images = 2*images.astype(np.float)/255. - 1
  N = labels.shape[0]
  #One-hot
  labels_tmp = np.zeros((N, 10))
  labels_tmp[np.arange(N), labels] = 1
  labels = labels_tmp
  return images, labels

images, labels = get_data()
images_test, labels_test = get_data('../input/data_batch_2')
N = labels.shape[0]

img = tf.placeholder(tf.float32, [None, 32, 32, 3])
x = tf.image.resize_bilinear(img, [224, 224])
y = tf.placeholder(tf.int32, [None, 10])

with slim.arg_scope(vgg.vgg_arg_scope()):
  logits, _ = vgg.vgg_16(x, num_classes=10)

print(map(lambda x: x.name, tf.trainable_variables()))


#Loss and initialize
loss = tf.contrib.losses.softmax_cross_entropy(logits, y)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#RUN
batch_size = 16
for i in range(100000):
  batch_ind = np.random.choice(N, batch_size)
  img_batch, label_batch = images[batch_ind], labels[batch_ind]
  loss_val, pred, _ = sess.run([loss, logits, train_op], {img: img_batch, y: label_batch})
  print('Loss:', loss_val, (pred.argmax(1) == label_batch.argmax(1)).mean())
  if i%100==0:
    batch_ind = np.random.choice(labels_test.shape[0], batch_size)
    img_batch, label_batch = images_test[batch_ind], labels_test[batch_ind]
    loss_val, pred = sess.run([loss, logits], {img: img_batch, y: label_batch})
    print('\n\tTEST Loss:', loss_val, (pred.argmax(1) == label_batch.argmax(1)).mean())
