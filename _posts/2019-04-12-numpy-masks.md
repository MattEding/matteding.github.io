## NumPy - Masks

In computer science, a mask is a bitwise filter for data.
Just as a real mask only lets parts of a face show through, masks only allow certain parts of data to be accessed.
Wherever a mask is `True`, we can extract corresponding data from a data structure.

![Mask](/images/mask.gif)

#### Reassignment
Masking NumPy arrays allows in-place assignment of the original array when indexing.
This way we can manipulate the values of the original array in a targeted manner.
```python
In [1]: import numpy as np

In [2]: arr = np.array([4, 88, 7, 12, 4, 9, 52, 4]).astype(float)

In [3]: outliers = (arr > 50)  # create the mask

In [4]: outliers
Out[4]: array([False,  True, False, False, False, False,  True, False])

In [5]: arr[outliers]  # select values from original where mask is True
Out[5]: array([88., 52.])

In [6]: arr[outliers] = np.nan  # in-place assignment

In [7]: arr
Out[7]: array([ 4., nan,  7., 12.,  4.,  9., nan,  4.])
```


#### Bitwise Operators
NumPy supports logical operators for boolean arrays.
We can combine different masks together to create new filters.

- `&` = [AND](https://en.wikipedia.org/wiki/Logical_conjunction): both `True`
- `|` = [OR](https://en.wikipedia.org/wiki/Logical_disjunction): at least one `True`
- `^` = [XOR](https://en.wikipedia.org/wiki/Exclusive_or): exactly one `True`
- `~` = [NOT](https://en.wikipedia.org/wiki/Negation): swaps `True` and `False`

```python
In [8]: bln_1 = np.array([True, True, False, False])

In [9]: bln_2 = np.array([True, False, True, False])

In [10]: bln_1 | bln_2
Out[10]: array([ True,  True,  True, False])

In [11]: bln_1 & bln_2
Out[11]: array([ True, False, False, False])

In [12]: bln_1 ^ bln_2
Out[12]: array([False,  True,  True, False])

In [13]: ~bln_1
Out[13]: array([False, False,  True,  True])
```

#### Querying
Moreover masks can be used to search data structures for information you need at hand.
Since Pandas is build upon NumPy, we can use masking to make queries on strings easily.

```python
In [14]: import pandas as pd

In [15]: string = ('NumPy is the building block of the Pandas library. '
                   'Masking works on Series, DataFrames, and N-dimensional Arrays.')

In [16]: words = pd.Series(string.split())

In [17]: lengthy = (words.str.len() >= 8)

In [18]: capitalized = words.str.istitle()

In [19]: words[lengthy]  # select long strings
Out[19]:
3          building
8          library.
13      DataFrames,
15    N-dimensional
dtype: object

In [20]: words[~lengthy & capitalized]  # select short, capitalized strings
Out[20]:
7      Pandas
9     Masking
12    Series,
16    Arrays.
dtype: object
```

While masks won't necessarily make you a crime-fighting vigilante, it will make you a superhero programmer.
So go out there and combat for better code!

```python
In [21]: exit()  # the end :D
```

*Code used to create the above animation is located at [my GitHub](https://github.com/MattEding/Python-Article-Resources/tree/master/masks).*
