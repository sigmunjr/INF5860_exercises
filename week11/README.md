# inf5860_uke11

Get the code for the exercises in:

    git clone https://github.com/sigmunjr/inf5860_exercises.git

If you have this repository all ready, you can update with:

    git pull

This weeks exercises is almost the same as week 1, only this time you will do them with tensorflow. You can use the built in convolution operator
tf.nn.conv2d from tensorflow. You should also try to build your own, simple convolutional neural network in CNN.py, but for this exercise, we have no test code.

In each of the files *indexing.py*, *math_operations.py* and *konvolusjon.py*, there is exercises to learn indexing, mathematical operations and
convolutions with the help of *python* and *tensorflow*.
In each of the files, there are a number of functions e.g. math1, math2 etc. You should implement these functions and make them return the correct result.
The target for each functions is written under the definition of each function e.g:


```python
def math1(mat):
  """
  Square each value in mat separately
  """
  return mat
```

This exercises can be solved by changing the function to this:

```python
def math1(mat):
  """
  Square each value in mat separately
  """
  return mat**2
```

To test if you have solved the exercises correctly, you can run:

    $ nosetests

from the folder week11. Then the files with filenames starting with test_ run, and your output will tell you if you solved the exercises correctly.
To test a single exercise you can e.g. run:

    $ nosetests test_indexing.py:test_index1

To test the index1 function, in the file indexing.py. You can also read the test-code, i e.g. *test_indexing.py* to better understand the exercise,
but do not alter the test code.

**Løsningene finner du ved**

    $ git clone https://github.com/sigmunjr/inf5860_solutions.git
