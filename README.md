# AlgZoo: uninterpreted models with fewer than 1,500 parameters

A model zoo of tiny RNNs and transformers trained to perform simple algorithmic tasks. These are intended to serve as a collection of "model organisms" for exhaustive mechanistic interpretability. For further explanation, see the blog post:

[**AlgZoo: uninterpreted models with fewer than 1,500 parameters**](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters)

## Installation

Recommended: Python 3.10+, pip 22.3+ (for editable installs), PyTorch 2.4.1+.

Example:
```
git clone git@github.com:alignment-research-center/alg-zoo.git
pip install -e alg-zoo
```

Check you can load the 432-parameter example 2nd argmax model:
```
from alg_zoo import example_2nd_argmax
model = example_2nd_argmax()
```

## Model families

The zoo contains four families of models trained on different algorithmic tasks. Models are stored in the publicly-accessible GCS folder `gs://arc-ml-public/alg/zoo`, and can be loaded using the appropriate function imported from `alg_zoo`, as detailed below.

### 2nd argmax

The 2nd argmax models are RNNs trained to find the position of the second-largest number in a sequence. They can be loaded using:

```
zoo_2nd_argmax(hidden_size, seq_len)
```

The available models and their accuracies are as follows:

![](images/zoo_2nd_argmax.svg)

### Argmedian

The argmedian models are RNNs trained to find the position of the middle number in an odd-length sequence. They can be loaded using:

```
zoo_argmedian(hidden_size, seq_len)
```

The available models and their accuracies are as follows:

![](images/zoo_argmedian.svg)

### Median

The median models are RNNs trained to output the median of a sequence. They can be loaded using:

```
zoo_median(hidden_size, seq_len)
```

The available models and their root mean squared errors are as follows:

![](images/zoo_median.svg)

### Longest cycle

The longest cycle models are transformers trained to count the length of the longest cycle of function *f*: {0, …, *n*−1} → {0, …, *n*−1}, i.e., the largest integer *k* such that for some *x*, the sequence *x*, *f*(*x*), *f*(*f*(*x*)), … returns to *x* for the first time after *k* steps. (We subtract 1 from the result to obtain something that lies in {0, …, *n*−1}.) They can be loaded using:

```
zoo_longest_cycle(hidden_size, seq_len)
```

The available models and their accuracies are as follows:

![](images/zoo_longest_cycle.svg)

## Additional models

Model checkpoints were saved after 0, 2^20, 2^21, ..., 2^28 and 2^29 out of 2^30 training sequences. These partially-trained models can be loaded using the `n_seqs` argument.

The models in the zoo are the best of 5 random seeds, since local minima can be hard to avoid for tiny models. The model with a particular seed from 0 to 4 can be loaded using the `seed` argument.

The handcrafted 2nd argmax models for sequence lengths 2, 3 and 10 discussed in the [blog post](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters) can be loaded using: `handcrafted_2nd_argmax(seq_len)`

The trained 2nd argmax model with sequence length 10 discussed in the [blog post](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters) is not officially part of the zoo, but can be loaded using: `example_2nd_argmax()`

## Training

The models in the zoo were trained using `train(task_name, hidden_size=hidden_size, seq_len=seq_len, seed=seed)`, with all other arguments set to their default values. Log files for these training runs can be found in the GCS folder.

Models were trained using PyTorch version `2.9.1+cu128` on an A100 (80G). It may be possible to bitwise reproduce these training runs using a similar setup, but this is not guaranteed to work.

The default value of the `n_train` argument is very large to maximize performance, but high-performance models can still be obtained using much lower values.

