This branch implements Google NIC model (O. Vinyals, A. Toshev, S. Bengio, and D. Erhan. Show and tell: A
neural image caption generator. In CVPR, 2015.)

The requirements of tools are:

| tool name   | version |
|:------------|:--------|
| pycocotools | 2.0.4   |
| torch       | 1.10.2  |
| nltk        | 3.6.5   |
| numpy       | 1.20.3  |
| pandas      | 1.3.4   |
| json5       | 0.9.6   |
| torchvision | 0.11.3  |

Using ```python evaluate_flickr30k.py``` can train the model and get the result. The result of this model with default
parameters are:

| Metric  | Score |
|:--------|:------|
| Bleu_1  | 0.476 |
| Bleu_2  | 0.272 |
| Bleu_3  | 0.152 |
| Bleu_4  | 0.088 |
| Meteor  | 0.135 |
| Rouge_L | 0.353 |
| CIDEr   | 0.211 |

The default parameters are:

| Parameter     | Value |
|:--------------|:------|
| nmin          | 50    |
| batch_size    | 64    |
| deterministic | True  |
| num_workers   | 2     |
| hidden_size   | 512   |
| max_length    | 15    |
| nepoch        | 15    |
| lr            | 0.001 |

with 32 batch_size:

| Metric  | Score |
|:--------|:------|
| Bleu_1  | 0.480 |
| Bleu_2  | 0.280 |
| Bleu_3  | 0.160 |
| Bleu_4  | 0.094 |
| Meteor  | 0.137 |
| Rouge_L | 0.357 |
| CIDEr   | 0.299 |



When evaluating models, ```pycocoeval``` are adopted.

notes:

The training set is flickr30k and the test set is COCO val2017. The training process
uses early stopping, and it stops at the 11th epoch. In the evaluation process, I import pycocoeval
code and delete SPICE metric due to unsolved bugs.