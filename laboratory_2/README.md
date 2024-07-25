# I used to hate NLP...

Let's now focus on NLP by working on some basic NLP tasks. [Hugging Face](https://huggingface.co/) Face seems to be the industry standard for NLP applications. However, I don't like it at all and try to avoid using it as much as possible. Specifically:

- I coded my own Trainer class, taking inspiration from PyTorch Lightning.
- I developed my own DistilBERT model, ensuring compatibility with Hugging Face weights.
- Datasets are good old fashioned Dataset classes from PyTorch. Hugging Face was only used to download the raw data and maybe perform some light preprocessing.
- Tokenizers, instead, are the ones provided by Hugging Face.

The main goals here are:

- Playing a bit with a pretrained GPT2 model and see what it generates.
- Perform text classification using DistilBERT.
- Implement a QA model.

## (dumb) HAL9000

Run the ```HAL9000.py``` script to have a chat with a GPT2 model. Don't expect too much however...

![](/laboratory_2/doc/HAL_not_being_HAL.gif)



## Classification
The classification pipeline follows a well-known approach that involves taking the logit of the [CLS] token produced by the BERT model (DistilBERT in this case), passing it through a classification head which projects it from the transformer dimensionality to a one-dimensional label, and applying the cross-entropy loss. The pipeline can be seen in the following image.

![](/laboratory_2/doc/classification.png)

Two datasets were used for this task:

- The IMDB dataset is for binary sentiment classification, containing substantially more data than previous benchmark datasets. It provides 25,000 highly polar movie reviews for training and another 25,000 for testing.
- The Hyperpartisan News Detection dataset involves determining whether a news article exhibits hyperpartisan argumentation, which means blind, prejudiced, or unreasoning allegiance to one party, faction, cause, or person. This dataset is small, with only 645 documents, and an 80/20 split was used for training and testing.

While the IMDB dataset is easier (simply counting the number of positive versus negative words in a review provides a strong baseline), the Hyperpartisan dataset poses a much subtle problem that is challenging to solve without a pretrained model. Furthermore, we are currently using the standard approach for dealing with sequences longer than 512 tokens (after tokenization): truncation. This is not a significant issue for IMDB, which has only 13.2% of observations exceeding this value, whereas Hyperpartisan has around 53%.

Results can be see below:

<center>

|     |IMDB|Hyperpartisan|
|:----:|:----:|:----:|
|Accuracy| 92.42%  |   86.76%    |

</center>

## Question Answering

Multiple choice question answering in natural language processing is a task where a system is given a question and several answer choices, and it must select the correct answer from the provided options. This task involves understanding the question, evaluating the answer choices, and selecting the most appropriate response based on the context and content of the question. It tests the system's comprehension and reasoning abilities, often requiring a combination of information retrieval, knowledge representation, and inference skills.

For this task two dataset were used:

- [CommonsenseQA](https://aclanthology.org/N19-1421/): This dataset is designed to evaluate and advance the capabilities of natural language understanding systems by focusing on commonsense knowledge. It includes 12,247 multiple-choice questions that require drawing upon commonsense knowledge to be answered correctly. To ensure the complexity and difficulty of the questions, the dataset includes distractors.

- Wikihop: WikiHop is a multi-hop question-answering dataset. Instances in WikiHop consist of: a question, answer candidates (ranging from two candidates to 79 candidates), supporting contexts (ranging from three paragraphs to 63 paragraphs), and the correct answer. The objective here is to select the correct answer from the list of candidates that responds to the query question based only on the support documents. 89.39% of instances has a length of question plus answers plus supports that is bigger than 512 tokens.

### Data preparation
Taking inspiration from the [Longformer](https://arxiv.org/abs/2004.05150) paper both the tokenizer and the model were a bit twicked to solve this task.


#### CommonsenseQA

To prepare the data for input to DistilBERT I first tokenize the question and answer candidates using DistilBERT wordpiece tokenizer. Then we concatenate the question and answer candidates with special tokens as ```[q] question [/q] [ent] candidate1 [/ent] ... [ent] candidateN [/ent].``` The special tokens ```[q], [/q], [ent], [/ent]``` were added to the DistilBERT vocabulary and randomly initialized before finetuning.

<center>

![](/laboratory_2/doc/QA_inputs.png)

</center>

For prediction, a linear layer is attached to each ```[ent]``` that outputs a single logit, average over all logits for each candidate across the chunks, apply a softmax and use the cross entropy loss with the correct answer candidate.

#### Wikihop

Note that for Wikihop dataset the model used is a DistilRoBERTa.

Question and candidates are processed in the same way as CommonsenseQA. The contexts are also concatenated using RoBERTa’s document delimiter tokens as separators: ```</s> context1 </s> ... </s> contextM </s>```

<center>

![](/laboratory_2/doc/Wikihop.png)

</center>

After preparing the input data, we compute activations from the top layer of the model as follows. We take the question and answer candidates and concatenate them to as much context as possible up to the model sequence length (512 for RoBERTa, 4,096 for Longformer), run the sequence through the model, collect the output activations, and repeat until all of the context is exhausted. Then all activations for all chunks are concatenated into one long sequence. For prediction, we attach a linear layer to each [ent] that outputs a single logit, average over all logits for each candidate across the chunks, apply a softmax and use the cross entropy loss with the correct answer candidate.

### Model

![](/laboratory_2/doc/QA_model.png)



### How to run question answering on Wikihop

1. Download the Wikihop dataset from this [link](https://data.niaid.nih.gov/resources?id=zenodo_6407402)
2. Extract only the wikihop dataset directory and place it into ```/laboratory_1/data```
3. Run the ```wikihop.py``` script for the data preprocessing.
4. Once you have obtained the preprocessed .json files you can run the ```QA_wikihop.py```

To run the script, use the following command:

```python3 QA_wikihop.py [arguments]```

- `--train_data`: path to the train dataset (default: "laboratory_2/data/wikihop/train.tokenized.json")
- `--test_data`: path to the test dataset (default: "laboratory_2/data/wikihop/dev.tokenized.json")
- `--lr`: Learning rate (default: 0.1)
- `--epochs`: Number of epochs to train (default: 14)
- `--log`: Enables logging of the loss and accuracy metrics to Weights & Biases (default: False)
- `--project_name`: Name of the W&B project (default: "WikihopQA")


### How to run question answering on CommonsenseQA

To run the script, use the following command:

```python3 QA_commonsense.py [arguments]```

- `--lr`: Learning rate (default: 0.1)
- `--epochs`: Number of epochs to train (default: 14)
- `--log`: Enables logging of the loss and accuracy metrics to Weights & Biases (default: False)
- `--project_name`: Name of the W&B project (default: "WikihopQA")


### Results

<center>

|     |Wikihop|CommonsenseQA|
|:----:|:----:|:----:|
|Accuracy| 64.59% |   45.11%   |

</center>

CommonsenseQA results seem very low but if we check the [leaderboard](https://www.tau-nlp.org/csqa-leaderboard2) we see that a BERT-base model achives 52.6% accuracy. Considering that we are using a much smaller model the results are partially justified. Similar thing can be said for the Wikihop results where a standard RoBERTa-base model achives _only_ 72.4%