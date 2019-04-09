## Discrete Differential Evolution Text Summarizer
### Part 1

I took it upon myself to implement a new text summarizer in Python. While looking
into NLP tasks I stumbled upon evolutionary algorithms and thought it would be
awesome to program one of them from scratch.

For those of you who are unfamiliar  with evolutionary algorithms, they try to
emulate natural selection in the real world. They start with a random population
from which you create new offspring by mixing data from members of the parent
population. Then using a fitness function, you select the best individuals from
both the parents and offspring to propagate to the next generation. Furthermore,
a mutation step can occur which alters the data individuals carry to keep the
population from becoming stagnant (stuck in local minima). This process is
repeated until an optimal solution has converged.

Some nomenclature used in this writing:
- discrete differential evolution (dde): The algorithm component to make new offsring.
- document (doc): A collection of tokenized text. We will use a collection of sentences.
- population (pop): The group of individuals for each generation.
- chromosome (chrom): An individual in the population. In this case sentences.
- gene: The data of a chromosome. This will represent words used in the entire document.

So how is this applied to text summarization? Well, first you have to prepare
the text into numerical data to work with. This is achieved by splitting the text
into sentences, and then you make a set of distinct words that appear in each.
In my implementation we use Scikit-Learn's CountVectorizer and cast it into a bool
dtype to reflect that we only care it a word appears rather than the number of
times it shows.
```python
count_vec = CountVectorizer(stop_words='english')
document = (count_vec.fit_transform(sentence_tokens)
                     .toarray()
                     .astype(bool))
```
Using this document, we can compare the similarity of any two sentences with
Jaccard similarity. This is defined as the intersection divided by the union of
two sets. We leverage numpy bitwise operators for these set operations.
```python
def jaccard_similarity(a, b):
  intersection = np.sum(a & b)
  union = np.sum(a | b)
  return intersection / union
```
As an example to illustrate its use with unique words from sentences:  
__Sentence A:__ Python is a dynamic language.  
__Sentence B:__ C++ is a compiled language.  
__Intersection:__ {is, a, language}  
__Union:__ {python, is, a, dynamic, language, c++, compiled}  
__Jaccard:__ 3 / 7 = 0.486  

Our goal is to partition the sentences in the document into clusters, from which
we will pick representative sentences from each to use in the summary. To achive
this, we will introduce cohesion (closeness within a cluster) and separation (how
far apart different clusters are). From the papers I researched we can define
them as follows:
```python
# to maximize
def cohesion(chrom, doc):
    total = 0
    for p in np.unique(chrom):
        sents = doc[chrom == p]
        for sent_i, sent_j in itertools.combinations(sents, r=2):
            total += jaccard_similarity(sent_i, sent_j) / len(sents)
    return total

# to minimize
def separtion(chrom, doc):
    total = 0
    for p, q in itertools.combinations(np.unique(chrom), r=2):
        sents_p = doc[chrom == p]
        sents_q = doc[chrom == q]
        for sent_i, sent_j in itertools.product(sents_p, sents_q):
            total += jaccard_similarity(sent_i, sent_j) / (len(sents_p) * len(sents_q))
    return total
```
We aim to have a balance between the above functions. To do so we introduce the
sigmoid function.
```python
def sigmoid(x):
    return 1 / (1 + np.exp(x))

# to maximize
def cohesion_separation(chrom, doc):
    coh = cohesion(chrom, doc, sim)
    sep = separation(chrom, doc, sim)
    return (1 + sigmoid(coh)) ** sep
```
Now that we have the prerequisite tools, we will delve into the evolutionary
algorithm in Part 2, so stay tuned for more!

For the full implementation feel free to look at my [EvolveSumm](https://github.com/MattEding/EvolveSumm)
repository on GitHub.
