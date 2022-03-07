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

**final results with beam search:**

flickr8k, 500 small demo test set:

| Metric  | Score |
|:--------|:------|
|Bleu_1:| 0.653|
|Bleu_2: |0.490|
|Bleu_3: |0.356|
|Bleu_4: |0.249|
|METEOR: |0.224|
|ROUGE_L: |0.505|
|CIDEr: |0.656|

flickr8k test set with beam_size = 1 (greedy search):

| Metric  | Score |
|:--------|:------|
|Bleu_1: |0.642|
|Bleu_2: |0.472|
|Bleu_3: |0.336|
|Bleu_4: |0.236|
|METEOR: |0.222|
|ROUGE_L: |0.501|
|CIDEr: |0.604|

flickr8k test set with beam_size = 2:

| Metric  | Score |
|:--------|:------|
|Bleu_1: |0.418|
|Bleu_2: |0.226|
|Bleu_3: |0.118|
|Bleu_4: |0.059|
|METEOR: |0.139|
|ROUGE_L: |0.311|
|CIDEr: |0.190|

flickr8k test set with beam_size = 3:

| Metric  | Score |
|:--------|:------|
|Bleu_1: |0.308|
|Bleu_2: |0.131|
|Bleu_3: |0.054|
|Bleu_4: |0.020|
|METEOR: |0.107|
|ROUGE_L: |0.227|
|CIDEr: |0.097|

flickr8k test set with beam_size = 4:

| Metric  | Score |
|:--------|:------|
|Bleu_1: |0.258|
|Bleu_2: |0.092|
|Bleu_3: |0.032|
|Bleu_4: |0.010|
|METEOR: |0.096|
|ROUGE_L: |0.191|
|CIDEr: |0.074|

flickr8k test set with beam_size = 5:

| Metric  | Score |
|:--------|:------|
|Bleu_1: |0.228|
|Bleu_2: |0.071|
|Bleu_3: |0.021|
|Bleu_4: |0.005|
|METEOR: |0.088|
|ROUGE_L: |0.169|
|CIDEr: |0.060|

flickr8k test set with beam_size = 6:

| Metric  | Score |
|:--------|:------|
|Bleu_1: |0.211|
|Bleu_2: |0.057|
|Bleu_3: |0.015|
|Bleu_4: |0.003|
|METEOR: |0.085|
|ROUGE_L: |0.157|
|CIDEr: |0.054|