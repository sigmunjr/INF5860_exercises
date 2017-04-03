import tensorflow as tf
import numpy as np
from skimage.io import imread, imsave
from matplotlib import pyplot as plt



def test_convolution():
  from konvolusjon import convolution
  sess = tf.InteractiveSession()
  img = plt.imread('lena.png')
  img = tf.convert_to_tensor(img)

  out = convolution(img, tf.convert_to_tensor(np.arange(25).reshape((5, 5)).astype(np.float32)))
  out = out.eval()
  out -= out.min()
  out /= out.max()
  correct = plt.imread('convolution_lena.png')[:, :, :3]
  assert np.abs(out[10:-10, 10:-10] - correct[10:-10, 10:-10]).max() < 1e-2
  sess.close()

def test_sobel_filter():
  from konvolusjon import sobel_filter
  sess = tf.InteractiveSession()
  img = plt.imread('lena.png')
  img = tf.convert_to_tensor(img)

  out = sobel_filter(img)
  out = sess.run(out)
  out -= out.min()
  out /= out.max()
  correct = plt.imread('sobel_lena.png')[:, :, :3]
  print np.abs(out[10:-10, 10:-10] - correct[10:-10, 10:-10]).max()
  assert np.abs(out[10:-10, 10:-10] - correct[10:-10, 10:-10]).max() < 1e-2
  sess.close()

def test_blur_filter():
  from konvolusjon import blur_filter
  sess = tf.InteractiveSession()
  img = plt.imread('lena.png')
  out = blur_filter(img)
  out = out.eval()
  out -= out.min()
  out /= out.max()
  correct = plt.imread('blur_lena.png')[:, :, :3]
  assert np.abs(out[10:-10, 10:-10] - correct[10:-10, 10:-10]).max() < 1e-2
  sess.close()