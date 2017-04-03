import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt



def test_index1():
  from indexing import index1
  mat = np.arange(16).reshape((4, 4))
  mat = tf.convert_to_tensor(mat.astype(np.float32))
  sess = tf.InteractiveSession()
  assert (index1(mat).eval() == np.array([4, 5, 6, 7])).all()
  assert (index1(tf.transpose(mat)).eval() == np.array([1,  5,  9, 13])).all()


def test_index2():
  from indexing import index2
  mat = np.arange(16).reshape((4, 4))
  mat = tf.convert_to_tensor(mat.astype(np.float32))
  sess = tf.InteractiveSession()
  assert (index2(mat).eval() == np.array([1,  5,  9, 13])).all()
  assert (index2(tf.transpose(mat)).eval() == np.array([4, 5, 6, 7])).all()


def test_index3():
  from indexing import index3
  mat = np.arange(16).reshape((4, 4))
  mat = tf.convert_to_tensor(mat.astype(np.float32))
  sess = tf.InteractiveSession()
  assert (index3(mat).eval() == np.array([[5,  6], [9, 10]])).all()
  assert (index3(tf.transpose(mat)).eval() == np.array([[5,  6], [9, 10]]).T).all()


def test_index4():
  from indexing import index4
  mat = np.arange(16).reshape((4, 4))
  mat = tf.convert_to_tensor(mat.astype(np.float32))
  sess = tf.InteractiveSession()
  assert (index4(mat).eval() == np.array([[1,  3], [ 5,  7], [ 9, 11], [13, 15]])).all()
  assert (index4(tf.transpose(mat)).eval() == np.array([[4, 12], [5, 13], [6, 14], [7, 15]])).all()


def test_index5():
  from indexing import index5
  mat = np.arange(16).reshape((4, 4))
  mat = tf.convert_to_tensor(mat.astype(np.float32))
  sess = tf.InteractiveSession()
  assert (index5(mat).eval() == np.array([[3,  1], [7,  5], [11,  9], [15, 13]])).all()
  assert (index5(tf.transpose(mat)).eval() == np.array([[12,  4], [13,  5], [14,  6], [15,  7]])).all()


def test_index6():
  from indexing import index6
  mat = np.arange(16).reshape((4, 4))
  mat = tf.convert_to_tensor(mat.astype(np.float32))
  sess = tf.InteractiveSession()
  assert (index6(mat).eval() == np.array([[ 0,  1,  2,  3], [8,  9, 10, 11]])).all()
  assert (index6(tf.transpose(mat)).eval() == np.array([[ 0,  4,  8, 12], [2, 6, 10, 14]])).all()


def test_index7():
  from indexing import index7
  mat = np.arange(16).reshape((4, 4))
  mat = tf.convert_to_tensor(mat.astype(np.float32))
  sess = tf.InteractiveSession()
  assert (index7(mat).eval() == np.array([[False, False, False, False],
                                    [ True,  True,  True,  True],
                                    [ True,  True,  True,  True],
                                    [ True,  True,  True,  True]], dtype=bool)).all()
  assert (index7(tf.transpose(mat)).eval() == np.array([[False,  True,  True,  True],
       [False,  True,  True,  True],
       [False,  True,  True,  True],
       [False,  True,  True,  True]], dtype=bool)).all()


def test_index8():
  from indexing import index8
  mat = np.arange(16).reshape((4, 4))
  mat = tf.convert_to_tensor(mat.astype(np.float32))
  sess = tf.InteractiveSession()
  assert (np.sort(index8(mat).eval().ravel()) == np.array([4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])).all()
  assert (np.sort(index8(tf.transpose(mat)).eval().ravel()) == np.array([4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])).all()

def test_index9():
  from indexing import index9
  mat = np.arange(16).reshape((4, 4))
  mat = tf.convert_to_tensor(mat.astype(np.float32))
  sess = tf.InteractiveSession()
  assert index9(mat).eval() == 12
  assert index9(tf.transpose(mat)).eval() == 12
