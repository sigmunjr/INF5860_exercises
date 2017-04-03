import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time

def main():
  img = plt.imread('lena.png')
  plt.imshow(img)

  start = time.time()
  out1 = sobel_filter(img)
  out2 = blur_filter(img)
  print 'Calculation time:', time.time()-start, 'sec'
  plt.figure()
  plt.imshow(out1.mean(2), vmin=out1.min(), vmax=out1.max(), cmap='gray')
  plt.figure()
  plt.imshow(out2, vmin=out2.min(), vmax=out2.max())
  plt.show()


def convolution(image, kernel):
  """
  Write a general function to convolve an image with an arbitrary kernel.
  Both image and kernel are tensorflow tensors, return a tensorflow tensor
  """

  return image


def blur_filter(img):
  """
  Use your convolution function to filter your image with an average filter (box filter)
  with kernel size of 11. Img are a tensorflow tensor. Return a tensor as well.
  """
  k_size = 11
  kernel = np.zeros((k_size, k_size))

  #YOUR CODE

  kernel = tf.convert_to_tensor(np.array(kernel, dtype=np.float32))
  return convolution(img, kernel)


def sobel_filter(img):
  """
  Use your convolution function to filter your image with a sobel operator in the vertical direction.
  Img are a tensorflow tensor. Return a tensor as well.
  """
  kernel = np.zeros((3, 3))

  #YOUR CODE

  kernel = tf.convert_to_tensor(np.array(kernel, dtype=np.float32))
  return convolution(img, kernel)


if __name__ == '__main__':
  main()