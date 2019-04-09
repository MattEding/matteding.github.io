## Discrete Differential Evolution Text Summarizer
### Part 2

In Part 1, we introduced metrics to use for determining how similar
sentences are using Jaccard, and ways to measure how well clusters are
distinguished from one another. Now we will go over the process to create
the summary using DDE.  

First, a random population is generated where each chromosome is made into a
partition of clusters. Each cluster represents a topic a given sentence
is assigned.

```python
#: make a chromosome that is a random partition with each cluster
clusters = np.arange(summ_len)
chrom = np.full(len(document), -1)
#: ensure that each cluster is accounted for at least once
idxs = np.random.choice(np.arange(len(chrom)), summ_len, replace=False)
chrom[idxs] = np.random.permutation(clusters)
#: fill rest randomly
idxs = (chrom == -1)
chrom[idxs] = np.random.choice(clusters, np.sum(idxs))
```

For example if there were 10 sentences and we want 3 sentence
summary, a possible chromosome in the population could be:  
__Sentence:__ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  
__Chromosome:__ [0, 2, 2, 1, 0, 1, 0, 2, 1, 1]  
From this we can see that topic 0 includes sentences 0, 4, and 6. Over time as
the algorithm evolves, these topic assignments will change in accordance to
how well their fitness metrics perform. After a desired amount of generations,
we will pick representative sentences from each cluster to use in the summary.  

With evolutionary algorithms, at each step we generate an offspring population
using the parent population. The way DDE works, we sample three distinct
chromosomes from the parent population (x_1, x_2, and x_3) to derive a new
chromosome. This is defined as x_1 + lambda(x_2 - x_3) where lambda is a scale
factor chosen by the user. To ensure it is discrete, we store the result in a
numpy array with an integer dtype and also use modulo by the number of clusters
to keep the number of desired summary sentences constant. Below is a vectorized
version to apply to the entire population array at once:  

```python
n = np.arange(len(pop))
s = frozenset(n)
#: get 3 distinct chromosomes that differ from i_th chromosome
idxs = np.array([np.random.choice(tuple(s - {i}), size=3, replace=False) for i in n])
chrom_1, chrom_2, chrom_3 = map(np.squeeze, np.split(pop[idxs], 3, axis=1))
#: discrete differential evolution
offspr = (chrom_1 + lambda_ * (chrom_2 - chrom_3)) % summ_len
```

For each parent/offspring pairing, we keep the better fit one according to
its `cohesion_separation` score (see part 1) by overriding the population array
in place. As this is the main bottleneck, use `multiprocessing` to utilize all
the cores on your machine.

```python
#: determine whether parents or offspring will survive to the next generation
with multiprocessing.Pool() as pool:
    fits = pool.map(cohesion_separation, itertools.chain(pop, offspr))
i = len(pop)
fit_pop, fit_off = fits[:i], fits[i:]
mask = fit_off > fit_pop
pop[mask] = offspr[mask]
```

Finally the new population is mutated according to a random state created anew
for each generation. For each chromosome, genes are randomly selected for
crossover--dubbed as such to mimic real life [chromosomal inversion](https://en.wikipedia.org/wiki/Chromosomal_crossover). Those selected genes have their
order reversed while other ones are kept in place. Below is the code to apply
this operation in vectorized form. Unfortunately I could not figure out a clearer
way to achieve this without explicit for loops.

```python
rand = np.random.random_sample(pop.shape)
mask = rand < sigmoid(pop)
#: inversion operator -> for each row reverse order of all True values
idxs = np.nonzero(mask)
arr = np.array(idxs)
sorter = np.lexsort((-arr[1], arr[0]))
rev = arr.T[sorter].T
pop[idxs] = pop[(rev[0], rev[1])]
```

This process is repeated for a fixed number of generations. After many iterations,
DDE gravitates towards optimal solutions as only most fit survives. In the 3rd
and final installment, we will go over how to extract the representative
sentences resulting from the evolutionary process to create the summary.

For the full implementation feel free to look at my [EvolveSumm](https://github.com/MattEding/EvolveSumm)
repository on GitHub.
