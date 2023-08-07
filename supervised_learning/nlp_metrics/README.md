# Natural Language Processing - Evaluation Metrics

[![](https://i.stack.imgur.com/MZygZ.png)](https://stackoverflow.com/questions/44324681/variation-in-bleu-score)

---
## Learning Objectives

### General

-   What are the applications of natural language processing?
-   What is a BLEU score?
-   What is a ROUGE score?
-   What is perplexity?
-   When should you use one evaluation metric over another?

## Requirements

### General

-   All files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
-   Files will be executed with `numpy` (version 1.15)
-   All code should follow the `pycodestyle` style (version 2.4)
-   Not allowed to use the `nltk` module

## Tasks

### 0. [Unigram BLEU score](https://github.com/wilstermanz/holbertonschool-machine_learning/blob/main/supervised_learning/nlp_metrics/0-uni_bleu.py)

Write the function `def uni_bleu(references, sentence):` that calculates the unigram BLEU score for a sentence:

-   `references` is a list of reference translations
    -   each reference translation is a list of the words in the translation
-   `sentence` is a list containing the model proposed sentence
-   Returns: the unigram BLEU score

```
$ cat 0-main.py
#!/usr/bin/env python3

uni_bleu = __import__('0-uni_bleu').uni_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))
$ ./0-main.py
0.6549846024623855
$

```

**Repo:**

-   GitHub repository: `holbertonschool-machine_learning`
-   Directory: `supervised_learning/nlp_metrics`
-   File: `0-uni_bleu.py`

### 1. [N-gram BLEU score](https://github.com/wilstermanz/holbertonschool-machine_learning/blob/main/supervised_learning/nlp_metrics/1-ngram_bleu.py)

Write the function `def ngram_bleu(references, sentence, n):` that calculates the n-gram BLEU score for a sentence:

-   `references` is a list of reference translations
    -   each reference translation is a list of the words in the translation
-   `sentence` is a list containing the model proposed sentence
-   `n` is the size of the n-gram to use for evaluation
-   Returns: the n-gram BLEU score

```
$ cat 1-main.py
#!/usr/bin/env python3

ngram_bleu = __import__('1-ngram_bleu').ngram_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(ngram_bleu(references, sentence, 2))
$ ./1-main.py
0.6140480648084865
$

```

**Repo:**

-   GitHub repository: `holbertonschool-machine_learning`
-   Directory: `supervised_learning/nlp_metrics`
-   File: `1-ngram_bleu.py`

### 2. [Cumulative N-gram BLEU score](https://github.com/wilstermanz/holbertonschool-machine_learning/blob/main/supervised_learning/nlp_metrics/2-cumulative_bleu.py)

Write the function `def cumulative_bleu(references, sentence, n):` that calculates the cumulative n-gram BLEU score for a sentence:

-   `references` is a list of reference translations
    -   each reference translation is a list of the words in the translation
-   `sentence` is a list containing the model proposed sentence
-   `n` is the size of the largest n-gram to use for evaluation
-   All n-gram scores should be weighted evenly
-   Returns: the cumulative n-gram BLEU score

```
$ cat 2-main.py
#!/usr/bin/env python3

cumulative_bleu = __import__('1-cumulative_bleu').cumulative_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(cumulative_bleu(references, sentence, 4))
$ ./2-main.py
0.5475182535069453
$

```

**Repo:**

-   GitHub repository: `holbertonschool-machine_learning`
-   Directory: `supervised_learning/nlp_metrics`
-   File: `2-cumulative_bleu.py`
