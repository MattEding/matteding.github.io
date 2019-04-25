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
- Other Libraries
  1. [Pandas](#pandas)
  - [Scikit-Learn](#scikit-learn)
  - PyData Sparse
- [Final Thoughts](#final-thoughts)


## Introduction
A [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix) is a matrix that has a value of 0 for most elements.
If the ratio of <b>N</b>umber of <b>N</b>on-<b>Z</b>ero ([NNZ](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.spmatrix.getnnz.html)) elements to the size is less than 0.5, the matrix is sparse.
While this is the mathematical definition, I will be using the term sparse for matrices with only NNZ elements and dense for matrices with all elements.

![Sparse vs Dense](/images/sparse_dense.gif)

Storing information about all the 0 elements is inefficient, so we will assume unspecified elements to be 0.
Using this scheme, sparse matrices can perform faster operations and use less memory than its corresponding dense matrix representation, which is especially important when working with large data sets in data science.

Today we will investigate all of the different implementations provided by the [SciPy sparse package](https://docs.scipy.org/doc/scipy/reference/sparse.html).
This implementation is modeled after `np.matrix` opposed to `np.ndarray`, thus is restricted to 2-D arrays and having quirks like `A * B` doing matrix multiplication instead of element-wise multiplication.
There are other implementations such as [PyData's sparse library](https://github.com/pydata/sparse) that provide interfaces like `np.ndarray`, but note that it has a limited menu of sparse formats.

```python
In [0]: from scipy import sparse

In [1]: import numpy as np

In []: spmatrix = sparse.random(10, 10)

In [161]: spmatrix
Out[161]:
<10x10 sparse matrix of type '<class 'numpy.float64'>'
        with 1 stored elements in COOrdinate format>

In []: spmatrix.nnz / np.product(spmatrix.shape)  # sparsity
Out[]: 0.01
```


## Construction Matrices
Different sparse formats have their strengths and weaknesses.
A good starting point is looking at formats that are efficient for constructing these matrices.
Typically you would start with one of these forms and then convert to another when ready to do calculations.


### Coordinate Matrix
Perhaps the simplest sparse format to understand is the <b>COO</b>rdinate ([COO](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html)) format.
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


### Dictionary of Keys Matrix

<b>D</b>ictionary <b>O</b>f <b>K</b>eys ([DOK](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dok_matrix.html))
is very much like COO except that it subclasses `dict` to store coordinate-data information as key-value pairs.
Since it uses a [hash table](https://en.wikipedia.org/wiki/Hash_table) as storage, identifying values at any given location has constant lookup time.
Use this format if you need the functionality that come with builtin dictionaries, but be mindful that hash tables hog much more memory than arrays.

```python
In [35]: dok = sparse.dok_matrix((10, 10))

In [36]: dok[(3, 7)] = 42  # store value 42 at coordinate (3, 7)

In []: dok[(9, 5)]  # zero elements are accessible
Out[]: 0.0

In [151]: dok.keys() | dok.transpose().keys()  # union of key views
Out[151]: {(3, 7), (7, 3)}

In [37]: isinstance(dok, dict)
Out[37]: True
```

Be careful of potential problems using the methods inherited from `dict`; they don't always behave.

```python
In []: out_of_bounds = (999, 999)

In []: dok[out_of_bounds] = 1  # works as expected
IndexError: Index out of bounds.

In []: dok.setdefault(out_of_bounds)  # silently ignored...

In []: dok.toarray()  # ...until now
ValueError: row index exceeds matrix dimensions

In []: dok.pop(out_of_bounds)  # fix issue by removing bad point

In []: sparse.dok_matrix.fromkeys([..., ..., ...])  # don't get me started
TypeError: __init__() missing 1 required positional argument: 'arg1'
```


### Linked List Matrix

The most flexible format to insert data is through usage of <b>LI</b>nked <b>L</b>ist ([LIL](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html)) matrices.
Data can be set via [indexing and slicing](https://docs.scipy.org/doc/numpy/user/basics.indexing.html) syntax of NumPy to quickly populate the matrix.
In my opinion, LIL is the coolest sparse format for constructing sparse matrices from scratch.

![LIL Matrix](/images/lil.gif)

LIL stores information in `lil.rows` where each list represents a row index and the elements inside the list match columns.
In a parallel array, `lil.data`, the NNZ values are stored.
But unlike other sparse formats, these subarrays *cannot* be explicitly passed to the constructor; LIL matrices must be made from either an empty state or from existing matrices, dense or sparse.
Below is an illustration of various techniques to build up a LIL matrix.

```python
In []: lil = sparse.lil_matrix((6, 5), dtype=int)

In []: lil[(0, -1)] = -1  # set individual point

In []: lil[3, (0, 4)] = [-2] * 2  # set two points

In []: lil.setdiag(8, k=0)  # set main diagonal

In []: lil[:, 2] = np.arange(lil.shape[0]).reshape(-1, 1) + 1  # set entire column

In []: lil.toarray()
Out[]:
array([[ 8,  0,  1,  0, -1],
       [ 0,  8,  2,  0,  0],
       [ 0,  0,  3,  0,  0],
       [-2,  0,  4,  8, -2],
       [ 0,  0,  5,  0,  8],
       [ 0,  0,  6,  0,  0]])
```

So what's the drawback...? Well, it utilizes [jagged arrays](https://en.wikipedia.org/wiki/Jagged_array) under the hood which requires `np.dtype(object)`.
This costs a lot more memory than a rectangular array, so if the data is big enough, you may be forced to work with COO instead of LIL.
In short, LIL is mostly offered as a convenience, albeit an awesome one at that.

```python
In [189]: lil.rows
Out[189]:
array([list([0, 2, 4]), list([1, 2]), list([2]), list([0, 2, 3, 4]),
       list([2, 4]), list([2])], dtype=object)

In []: lil.data[:, np.newaxis]  # expose jagged structure
Out[]:
array([[list([8, 1, -1])],
       [list([8, 2])],
       [list([3])],
       [list([-2, 4, 8, -2])],
       [list([5, 8])],
       [list([6])]], dtype=object)
```

As an aside, Linked List Matrix is a misnomer since it does *not* actually use [linked lists](https://en.wikipedia.org/wiki/Linked_list) behind the scenes!
LIL actually uses Python's `list` which is actually a [dynamic array](https://en.wikipedia.org/wiki/Dynamic_array), so it should really be called a List of List Matrix, in spite of what the documentation says.
(A missed opportunity to dub it LOL... :disappointed:)

```python
In [301]: sparse.lil.__doc__  # module docstring
Out[301]: 'LInked List sparse matrix class\n'
```


## Compressed Sparse Matrices
The formats described earlier are great for building sparse matrices, but they aren't as computationally performant than more specialized forms.
The reverse is true for compressed sparse matrices, which should be treated as read-only rather than write-only.
These are more difficult to understand, but with a little patience their structure can be grasped.


### Compressed Spare Row/Column

The <b>C</b>ompressed <b>S</b>parse <b>R</b>ow/<b>C</b>olumn ([CSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) and [CSC](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html)) formats h............

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
Second, these values represent the [start:stop] slice of the indices array, and their difference is the NNZ elements in each row.
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


### Block Sparse Row
<b>B</b>lock <b>S</b>parse <b>R</b>ow ([BSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html)) is like CSR but stores sub-matrices rather than scalar values at locations.

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

This implementation requires all the sub-matrices to have the same shape, but there are more generalized constructs with [block matrices](https://en.wikipedia.org/wiki/Block_matrix) that relax this constraint.
Generalized block matrices do not have their unique data structure in SciPy, but can be indirectly using COO via the `sparse.bmat` constructor function.

```python
A = np.arange(8).reshape(2, 4)  # can use arrays

T = np.tri(5, 4)

L = [[8] * 4] * 2  # can use lists

I = sparse.identity(4)  # can use sparse array

Z = sparse.coo_matrix((2, 3))  # zeros to create column gap

In [267]: sp.bmat([[   A,    Z,    L],
     ...:          [None, None,    I],
     ...:          [   T, None, None]], dtype=int)
Out[267]:
<11x11 sparse matrix of type '<class 'numpy.int64'>'
        with 33 stored elements in COOrdinate format>

In [268]: _.toarray()
Out[268]:
array([[0, 1, 2, 3, 0, 0, 0, 8, 8, 8, 8],
       [4, 5, 6, 7, 0, 0, 0, 8, 8, 8, 8],
       [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
```


## Diagonal Matrix
Perhaps the most specialized of the formats to store sparse data is the <b>DIA</b>gonal ([DIA](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.dia_matrix.html)) variant.
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


## Other Libraries

### Pandas
...

### Scikit-Learn
The machine learning powerhouse, [Scikit-Learn](https://scikit-learn.org/stable/), supports sparse matrices in many areas.
This is important since big data is where sparse matrices thrive (assuming it is sparse enough).
After all, who wouldn't want to have performance gains from these number-crunching algorithms?
It hurts having to wait on CPU intensive [SVMs](https://scikit-learn.org/stable/modules/svm.html?highlight=sparse), not to mention the possibility of not having dense arrays fitting into working memory!

#:TODO Change  
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


### PyData Sparse
...


## Final Thoughts
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
Furthermore if you are working with more logical operators, you will be forced to look elsewhere.
But all these concerns aside, hopefully sparse matrices can help "lighten" your load.



```python
In [45]: exit()  # the end :D
```

*Code used to create the above animation is located at [my GitHub](https://github.com/MattEding/NumPy-Articles/tree/master/sparse-matrix).*
