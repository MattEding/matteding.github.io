# Data Structures - Sparse Matrices

#### Table of Contents
1. [Introduction](#introduction)
- [Construction Matrices](#construction-matrices)
    1. [Coordinate Matrix](#coordinate-matrix)
    - [Linked List Matrix](#linked-list-matrix)
    - [Dictionary of Keys Matrix](#dictionary-of-keys-matrix)
- [Compressed Sparse Matrices](#compressed-sparse-matrices)
    1. Compressed Sparse Row
    - Compressed Sparse Column
    - Block Sparse Row
- [Diagonal Matrix](#diagonal-matrix)
- [Specialized Functions](#specialized-functions)
- [Scikit-Learn](#scikit-learn)
- [Final Thoughts](#final-thoughts)


## Introduction
A [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix) is a matrix that has a value of 0 for most elements.
If the ratio of __N__umber of __N__on-__Z__ero ([NNZ](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.getnnz.html))__ __ elements to the size is less than 0.5, the matrix is sparse.
While this is the mathematical definition, I will be using the term sparse for matrices with NNZ elements and dense for matrices with all elements.

![Sparse vs Dense](/images/sparse_dense.gif)

Storing information about all the 0 elements is inefficient, so we will assume unspecified elements to be 0.
Using this scheme, sparse matrices can perform faster operations and use less memory than a dense matrix representation, which is especially important when working with large data sets in data science.

Today we will investigate several different implementations provided by the [SciPy sparse package](https://docs.scipy.org/doc/scipy/reference/sparse.html).
This implementation is modeled after `np.matrix` opposed to `np.ndarray`, thus is restricted to 2-D arrays and `A * B` does matrix multiplication rather than of element-wise multiplication.
There are other variants in the works such as [PyData's sparse library](https://github.com/pydata/sparse) in that provides an interface like `np.ndarray`, albeit with limited sparse formats.

```python
In [0]: from scipy import sparse

In [1]: import numpy as np

In []: spmatrix = sparse.random(10, 10)

In []: spmatrix.nnz / np.product(spmatrix.shape)  # sparsity
Out[]: 0.01
```


## Construction Matrices
...

### Coordinate Matrix
Perhaps the simplest sparse format to understand is the __COO__rdinate ([COO](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html))__ __
format.
This variant uses vectors to store the element values and their coordinate positions.

![COO Matrix](/images/coo.gif)

```python
In [2]: row = [1, 3, 0, 2, 4]

In [3]: col = [1, 4, 2, 3, 3]

In [4]: data = [2, 5, 9, 1, 6]

In [5]: coo = sparse.coo_matrix((data, (row, col)), shape=(6, 7))

In [6]: print(coo)  # coordinate format
# (1, 1)        2
# (3, 4)        5
# (0, 2)        9
# (2, 3)        1
# (4, 3)        6

In [7]: coo.todense()  # coo.toarray() for ndarray instead
Out[7]:
matrix([[0, 0, 9, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0],
        [0, 0, 0, 6, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]])
```

The savings on memory consumption is quite substantial as the matrix size increases.
The overhead incurred from needing to manage the subarrays is a fixed cost for sparse matrices unlike the case for dense matrices.
A word of caution: only use sparse arrays if they are sufficiently sparse enough; it would be counterproductive storing a mostly nonzero array using several subarrays to keep track of position and data.

```python
In [8]: def memory_usage(coo):
   ...:    # data memory and overhead memory
   ...:    coo_mem = (sum(obj.nbytes for obj in [coo.data, coo.row, coo.col])
   ...:               + sum(obj.__sizeof__() for obj in [coo, coo.data, coo.row, coo.col]))
   ...:    print(f'Sparse: {coo_mem}')
   ...:    mtrx = coo.todense()
   ...:    mtrx_mem = mtx.nbytes + mtrx.__sizeof__()
   ...:    print(f'Dense: {mtrx_mem}')

In [9]: memory_usage(coo)
# Sparse: 480
# Dense: 448

In [10]: coo.resize(100, 100)

In [11]: memory_usage(coo)
# Sparse: 480
# Dense: 80112
```

### Linked List Matrix
...

### Dictionary of Keys Matrix
...


## Compressed Sparse Matrices
...

### Compressed Spare Row
The COO format is great for building sparse matrices, but is not as performant for math operations as more specialized forms.
The __C__ompressed __S__parse __R__ow/__C__olumn ([CSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) and [CSC](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html))__ __
formats help address this.

![CSR Matrix](/images/csr.gif)

```python
In [12]: indptr = np.array([0, 2, 3, 3, 3, 6, 6, 7])

In [13]: indices = np.array([0, 2, 2, 2, 3, 4, 3])

In [14]: data = np.array([8, 2, 5, 7, 1, 2, 9])

In [15]: csr = sparse.csr_matrix((data, indices, indptr))

In [16]: csr.todense()
Out[16]:
matrix([[8, 0, 2, 0, 0],
        [0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 7, 1, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 9, 0]])
```

Adjacent pairs of index pointers determine two things.
First, their position in the pointer array is the row number.
Second, these values represent the [start:stop] slice of the indices array.
Their difference is the NNZ elements in each row.
Using the pointers, look up the indices to determine the column for each element in the data.
CSC works exactly the same but has column based index pointers and row indices instead.

These compressed sparse formats are great as read-only for computation.
Beware though that CSR, CSC, and BSR are not suited for writing new data points.
If you are inserting new data points, consider COO, LIL, or DOK instead.

```python
In [17]: csr.resize(1000, 1000)

In [18]: %timeit csr @ csr  # comparing matrix multiplication
# 111 µs ± 3.66 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

In [19]: coo = csr.tocoo()

In [20]: %timeit coo @ coo
# 251 µs ± 8.06 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

In [21]: arr = csr.toarray()

In [22]: %timeit arr @ arr
# 632 ms ± 2.02 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```


## Diagonal Matrix
Perhaps the most specialized of the formats to store sparse data is the __DIA__gonal ([DIA](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.html))__ __ format.
It is best suited for data that appears along the diagonals of a matrix.

![DIA matrix](/images/dia.gif)

```python
In [25]: data = np.arange(15).reshape(3, -1) + 1

In [26]: offsets = np.array([0, -3, 2])

In [27]: dia = sparse.dia_matrix((data, offsets), shape=(7, 5))

In [28]: dia.toarray()
Out[28]:
array([[ 1,  0, 13,  0,  0],
       [ 0,  2,  0, 14,  0],
       [ 0,  0,  3,  0, 15],
       [ 6,  0,  0,  4,  0],
       [ 0,  7,  0,  0,  5],
       [ 0,  0,  8,  0,  0],
       [ 0,  0,  0,  9,  0]])
```

The data is stored in an array of shape (offsets) x  (width) where the offsets dictate the location of each row in the data array along diagonal.
Offsets are below or above the main diagonal when negative or positive respectively.

Note that if a row in the data matrix is cutoff, the excess elements can take any value (but they must have placeholders).

```python
In [29]: dia.data.ravel()[9:12] = 0  # replace cutoff data

In [30]: dia.data
Out[30]:
array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9,  0],
       [ 0,  0, 13, 14, 15]])

In [31]: dia.toarray()  # same array repr as earlier
Out[31]:
array([[ 1,  0, 13,  0,  0],
       [ 0,  2,  0, 14,  0],
       [ 0,  0,  3,  0, 15],
       [ 6,  0,  0,  4,  0],
       [ 0,  7,  0,  0,  5],
       [ 0,  0,  8,  0,  0],
       [ 0,  0,  0,  9,  0]])
```

#### Other Formats
There are other structures used to represent sparse matrices effectively, each having their unique advantages and disadvantages (see the documentation for details).

- __LI__nked __L__ist ([LIL](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html))
- __D__ictionary __O__f __K__eys ([DOK](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html))
- __B__lock __S__parse __R__ow ([BSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html))__ __

The LIL format is much like COO but supports slicing to help construct sparse matrices from scratch.

```python
In [32]: lil = sparse.lil_matrix((4, 9), dtype=int)

In [33]: lil[:, 6] = np.random.sample((4, 1)) * 10  # fill entire column at once

In [34]: lil.todense()
Out[34]:
matrix([[0, 0, 0, 0, 0, 0, 9, 0, 0],
        [0, 0, 0, 0, 0, 0, 9, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 0],
        [0, 0, 0, 0, 0, 0, 7, 0, 0]])
```

DOK is also very like COO except that it subclasses `dict` to store coordinate-data key-value pairs.

```python
In [35]: dok = sparse.dok_matrix((10, 10))

In [36]: dok[(3, 7)] = 42  # store value 42 at coordinate (3, 7)

In [37]: isinstance(dok, dict)
Out[37]: True
```

BSR is like CSR but stores sub-matrices rather than scalar values at locations.
This implementation requires all the sub-matrices to have the same shape, but there are more generalized constructs with [block matrices](https://en.wikipedia.org/wiki/Block_matrix) that relax this constraint.

```python
In [38]: ones = np.ones((2, 3), dtype=int)

In [39]: data = np.array([ones + i for i in range(4)])

In [40]: indices = [1, 2, 2, 0]

In [41]: indptr = [0, 2, 3, 4]

In [42]: bsr = sparse.bsr_matrix((data, indices, indptr))

In [43]: bsr.todense()
Out[43]:
matrix([[0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 3, 3, 3],
        [0, 0, 0, 0, 0, 0, 3, 3, 3],
        [4, 4, 4, 0, 0, 0, 0, 0, 0],
        [4, 4, 4, 0, 0, 0, 0, 0, 0]])
```

#### Specialized Functions
In addition to the multitude of formats, there is a plethora of functions specialized just for sparse matrices.
Use these functions whenever possible rather than their NumPy counterparts, otherwise speed performances will be compromised.

- [general functions](https://docs.scipy.org/doc/scipy/reference/sparse.html#functions)
  - scipy.sparse.save_npz
  - scipy.sparse.isspmatrix
  - scipy.sparse.hstack
  - ...
- [linear algebra](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
  - scipy.sparse.linalg.svds
  - scipy.sparse.linalg.inv
  - scipy.sparse.linalg.norm
  - ...
- [graph algorithms](https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html)
  - scipy.sparse.csgraph.dijkstra
  - scipy.sparse.csgraph.minimum_spanning_tree
  - scipy.sparse.csgraph.connected_components
  - ...

#### Scikit-Learn
The machine learning powerhouse, [Scikit-Learn](https://scikit-learn.org/stable/), supports sparse matrices in many areas.
This is important since big data is where sparse matrices thrive (assuming it is sparse enough).
After all, who wouldn't want to have performance gains from these number-crunching algorithms?
It hurts having to wait on CPU intensive [SVMs](https://scikit-learn.org/stable/modules/svm.html?highlight=sparse), not to mention the possibility of not having dense arrays fitting into working memory!

__#:TODO Change__  
CSR is the object type of Scikit-Learn's term-document matrices produced by its [text vectorizers](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text).
This is crucial for NLP since most words are used sparingly if at all.
Naively using a dense format might otherwise cause speed bottlenecks not to mention the possibility of not fitting in working memory.

```python
In [23]: from sklearn.feature_extraction.text import CountVectorizer

In []: bow = CountVectorizer().fit_transform(['sparse'])

In [24]: sparse.isspmatrix(bow)
Out[24]: True

In []: sparse.save_npz('bag_of_words.npz', bow)  # store for future use
```

Other areas where Scikit-Learn has the ability to output sparse matrices include:
- sklearn.preprocessing.OneHotEncoder
- sklearn.preprocessing.LabelBinarizer
- sklearn.feature_extraction.DictVectorizer

Moreover there are utilities that play well with sparse matrices such as [scalers](https://scikit-learn.org/stable/modules/preprocessing.html#scaling-sparse-data), a handful of [decompositions](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition), some [pairwise distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances), [train-test-split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split), and *many* estimators can predict and/or fit sparse matrices.
In short embrace their usage whenever possible to make your machine learning models more efficient.

#### Final Thoughts
Unfortunately [logical operators](https://docs.scipy.org/doc/numpy/reference/ufuncs.html#comparison-functions) are not directly supported for boolean sparse matrices.
Luckily it is not too difficult to implement `&` and `|`, but `~` is not doable because it would make a sparse matrix into a dense matrix.

```python
In [44]: class LogicalSparse(sparse.coo_matrix):
    ...:    def __init__(self, *args, **kwargs):
    ...:        super().__init__(*args, dtype=bool, **kwargs)  # leverage existing base class
    ...:
    ...:    def __and__(self, other):  # self & other
    ...:        return self.multiply(other)
    ...:
    ...:    def __or__(self, other):  # self | other
    ...:        return self + other
```

Keep in mind that while sparse matrices are are great tool, they are not necessarily a replacement for arrays.
If a matrix is not sufficiently sparse, the multitude of storage arrays behind the scenes will actually take up more resources than a regular dense array would.
Furthermore if you are working with more logical operators than AND or OR, you will be forced to look elsewhere.
But all these concerns aside, hopefully sparse matrices can help "lighten" your load.



```python
In [45]: exit()  # the end :D
```

*Code used to create the above animation is located at [my GitHub](https://github.com/MattEding/NumPy-Articles/tree/master/sparse-matrix).*
