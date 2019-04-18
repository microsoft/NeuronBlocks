# **NeuronBlocks** Tutorial

* [Installation](#Installation)
* [Quick Start](#Quick-Start)
* [How to Design Your NLP Model](#How-to-Design-Your-NLP-Model)
* [Model Zoo for NLP Tasks](#Model-Zoo-for-NLP-Tasks)
* [Advanced Usage](#Advanced-Usage)
* [FAQ](#FAQ)

## Installation

1. Clone this project. 
```bash
git clone https://github.com/Microsoft/NeuronBlocks.git
```
2. Install the python packages in requirements.txt by the following command:
```bash
pip install -r requirements.txt
```
Specially, we suggest you to install PyTorch by following the instruction of [PyTorch](https://pytorch.org/get-started/previous-versions/).
Note: **Tookit supports PyTorch version 0.4.1.**

## Quick Start

You can start by trying the given examples:
```bash
# train
python train.py --conf_path=model_zoo/demo/conf.json

# test
python test.py --conf_path=model_zoo/demo/conf.json

# predict
python predict.py --conf_path=model_zoo/demo/conf.json
```


## How to Design Your NLP Model

Tips: in the following instruction, PROJECT_ROOT denotes the project root of this project.
### Define the model configuration file

To training a neural network, you need to define your model architecture and some other settings to a configuration file, which is in JSON format. 

You can make your private folder in PROJECT_ROOT/model_zoo/ folder and put your model configuration in PROJECT_ROOT/model_zoo/your_folder. The data which used for model train/test/predict is put in PROJECT_ROOT/dataset.

Take the PROJECT_ROOT/model_zoo/demo/conf.json as an example(we make it more suitable for the usage explanation so that the model architecture might not be practical),the configuration is used for question answer matching task, which aims to figure out whether the answer of that query can be found in the passage.The data sample lies in PROJECT_ROOT/dataset/demo.

The architecture of the configuration file is:
- **inputs**. This part define the input configuration.
    - ***use_cache***. If use_cache is true, the toolkit would make cache at the first time so that we can accelerate the training process at the next time.
    - ***dataset_type***. Declare the task type here. Currently, we support classification, sequence_tagging, regression and mrc.
    - ***data_paths***.
        - *train_data_path*. [necessary for training] Data for training.
        - *valid_data_path*. [optional for training] Data for validation. During training, the toolkit would save the model which have the best performance on validation data. If you don't need a validation, just remove this node.
        - *test_data_path*. [necessary for training, test] Data for test. If valid_data_path is not defined, the toolkit would save the model which have the best performance on test data.
        - *predict_data_path*. [conditionally necessary for prediction] Data for prediction. When we are predicting, if predict_data_path is not declared, the toolkit will predict on test_data_path instead.
        - *pre_trained_emb*. [optional for training] Pre-trained embeddings.
    - ***pretrained_emb_type***. [optional, default: glove] We support glove, word2vec, fasttext now.
    - ***pretrained_emb_binary_or_text***. [optional, default: text] We support text and binary.
    - ***involve_all_words_in_pretrained_emb***. [optional, default: false] If true, all the words in the pretrained embedings are added to the embedding matrix.
    - ***add_start_end_for_seq***. [optional, default: True] For sequences in data or target, whether to add start and end tag automatically.
    - ***file_header***. [necesssary for training and test] This part defines the file format of train/valid/test data. For instance, the following configuration means there are 3 columns in the data, and we name the first to fifth columns as query_index, passage_index, label_index, respectively.
        ```json
        "file_header": {
          "query_index": 0,
          "passage_index": 1,
          "label_index":   2
        }
        ```
    - ***predict_file_header***. [conditionally necessary for prediction] This part defines the file format of prediction data. **If the file_header of predict_data is not consistent with file_header of train/valid/test data, we have to define "predict_file_header" for predict_data, otherwise conf[inputs][file_header] is applied to the prediction data by default.** Two file_headers are consistent if the indices of data column involved in conf[inputs][model_inputs]) are consistent.
        ```json
        "predict_file_header": {
          "query_index": 0,
          "passage_index": 1
        },
        ```
    - ***file_with_col_header***.[optional, default:False] If your train dataset has column name title, remember to set value to True. Otherwise, it may lead program error.
    - ***model_inputs***. The node is used for define model inputs. In this example task, there are two inputs: query and passage:
        ```json
        "model_inputs": {
          "query": [
            "query_index"
          ],
          "passage": [
            "passage_index"
          ]
        }
        ```
    - ***target***. [necessary for training and test, some part necessary for prediction] This node defines the target column in the train/valid/test data.The type of target is array because our tookit support multi-target tasks.
- **outputs**. This node defines the settings of path to save models and logs, as well as cache.
    - ***save_base_dir***. The directory to save model and logs.
    - ***model_name***. The model would be saved as save_base_dir/model_name.
    - ***train_log_name/test_log_name/predict_log_name***.
    - ***predict_fields***. A list to set up the fields you want to predict, such as prediction and confidence.
    - ***cache_dir***. The directory to save cache.
- **training_params**. We define the optimizer and training hyper parameters here.
    - ***optimizer***. 
        - *name*. We support all the optimizers defined in [torch.optim](http://pytorch.org/docs/0.3.1/optim.html?#module-torch.optim).
        - *params*. The optimizer parameters are exactly the same as the parameters of the initialization function of optimizers in [torch.optim](http://pytorch.org/docs/0.3.1/optim.html?#module-torch.optim).
    - ***use_gpu***. [default: True] Whether to use GPU if there is at least one GPU available.
    - ***batch_size***. Define the batch size here.
    - ***batch_num_to_show_results***. [necessary for training] During the training process, show the training progress every batch_num_to_show_results batches.
    - ***max_epoch***. [necessary for training] The maximum number of epochs to train.
    - ***valid_times_per_epoch***. [optional for training, default: 1] Define how many times to conduct validation per epoch. Usually, we conduct validation after each epoch, but for a very large corpus, we'd better validate multiple times in case to miss the best state of our model. The default value is 1.
- **architecture**. Define the model architecture. The node is a list to represent layers of a model. The supported layers of this tool are given on [Layers overview](http://10.177.74.200:8080/layers.html). 
    
    - ***Embedding layer***. The first layer of this example (as shown below) defines the embedding layer, which is composed of two types of embedding: "word" (word embedding). The name of embedding types can be defined as any other name. It denotes that the the columns related to "word" are query_index and passage_index. Lastly, the dimension of "word" are 300.
        ```json
        {
            "layer": "Embedding",
            "conf": {
              "word": {
                "cols": ["query_index", "passage_index"],
                "dim": 300
              }
             }
        }
        ```
    - Using block_zoo to design your model. You can choose layers in block_zoo to build your model follow the fomat:
        - *layer_id*.Customized name for one model layer
        - *layer*. The layer name in block_zoo  
        - *conf*. Each layer has their own config(you can find layer name and corresponding parameters in [Layers overview](http://10.177.74.200:8080/layers.html))
        - *inputs*. The layer_id which connect to this layer, the type of inputs must be array, because one layer can have multi-layer input.
    This is a BiLSTM layer example: 
        ```json
        {
            "layer_id": "query_1_before_ln",
            "layer": "BiLSTM",
            "conf": {
              "hidden_dim": 64,
              "dropout": 0
            },
            "inputs": ["query"]
        }
        ```
        To know more about supported layers and their configuration, please go to [Layers overview](http://10.177.74.200:8080/layers.html). For example, if we want to know the parameters of BiLSTM, we can find that there is a BiLSTM class and a BiLSTMConf class, the parameters of BiLSTM would be given at BiLSTMConf.
- **loss**. [necessary for training and test] Currently, we support all the loss functions offered by [PyTorch loss functions](http://pytorch.org/docs/0.3.1/nn.html#loss-functions). The parameters defined in configuration/loss/conf are exactly the same with the parameters of initialization function of loss functions in [PyTorch loss functions](http://pytorch.org/docs/0.3.1/nn.html#loss-functions). Additionally, we offer more options, such as [Focal Loss](https://arxiv.org/pdf/1708.02002.pdf), please refer to [Loss function overview](http://10.177.74.200:8080/losses.html).
        Specially, for classification tasks, we usually add a Linear layer to project the output to dimension of number of classes, if we don't know the #classes, we can use '-1' instead and we would calculate the number of classes from the corpus.
- **metrics**. 
    - For classification task, metrics supported: *auc, accuracy, f1, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall, weighted_f1, weighted_precision, weighted_recall* are supported. 
    - For sequence_tagging task, metrics supported: *seq_tag_f1 and accuracy*.
    - For regression task, metrics supported: *MSE, RMSE*.
    - For mrc task, metrics supported: *f1, em*.
    
    During validation, the toolkit select the best model according to the first metric.

*Tips: the [optional] and [necessary] mark means corresponding node in the configuration file is optional or necessary for training/test/prediction. If there is no mark, it means the node is necessary all the time. Actually, it would be more convenient to prepare a configuration file that contains all the configurations for training, test and prediction.*



## Visualize your model

A model visualizer is provided for visualization and configuration correctness checking, please refer to [MV_README.md](MV_README.md).

## Model Zoo for NLP Tasks

In Model Zoo, we provide a suite of NLP models for common NLP tasks, in the form of JSON configuration files. You can pick one of existing models (config files) in Model Zoo to start model training quickly, or build your own models by modifying the JSON configuration file to suit your specific task.


### Task 1: Text Classification
Text classification is a core problem to many applications like spam filtering, email routing, book classification. The task aim to train a classifier using labelled dataset containing text documents and their labels, which can be a web page, paper, email, reviewer etc.
- ***Dataset***
The [20 Newsgroups data set](http://qwone.com/~jason/20Newsgroups/) is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. Here is a list of the 20 newsgroups, partitioned (more or less) according to subject matter:
![](https://i.imgur.com/rHLITSi.png)

- ***Usage***
Note: PROJECT_ROOT denotes the project root of this project.

    1. run automatic downloading and preprocessing script.
    ```bash
    cd PROJECT_ROOT/dataset
    python get_20_newsgroups.py
    ```
    2. train text classification model.
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/text_classification/conf_text_classification_cnn.json 
    ```
     *Tips:can try different model by run different config file* 
- ***Result***
    

    | Model    | Accuracy |
    | -------- | -------- |
    | TextCNN  | 0.9610   |
    | BiLSTM+Attention|0.9707|
    *Tips: the model file and train log file can find in config file's outputs/save_base_dir after you finish trainning*


### Task 2: Question Answer Matching

Question answer matching is a crucial subtask of the question answering problem, with the aim of determining question-answer pairs are matched or not.

- ***Dataset***
[Microsoft Research WikiQA Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52419) is a publicly available set of question and sentence pairs, collected and annotated for research on open-domain question answer matching. WikiQA includes 3,047 questions and 29,258 sentences, where 1,473 sentences were labeled as answer sentences to their corresponding questions. More detail of this corpus can be found in the paper [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/).

- ***Usage***
Note: PROJECT_ROOT denotes the project root of this project.

    1. run automatic downloading script
    ```bash
    cd PROJECT_ROOT/dataset
    python get_WikiQACorpus.py
    ```
    2. train question answer matching model
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/question_answer_matching/conf_question_answer_matching_bilstm.json
    ```
     *Tips:can try different model by run different config file.* 
- ***Result***
    
    | Model    | AUC |
    | -------- | -------- |
    | CNN (WikiQA paper) | 0.7359 |
    | CNN-Cnt (WikiQA paper) | 0.7533 |
    | CNN (NeuronBlocks) | 0.7479 |
    | BiLSTM (NeuronBlocks) | 0.7673 |
    | BiLSTM+Attn (NeuronBlocks) | 0.7548 |
    
    *Tips: the model file and train log file can find in config file's outputs/save_base_dir after you finish trainning*

### Task 3: Question Natural Language Inference
Natural language inference (NLI) is a task that incorporates much of what is necessary to understand language, such as the ability to leverage world knowledge or perform lexico-syntactic reasoning. Given two sentences, a premise and a hypothesis, an NLI system must determine whether the hypothesis is implied by the premise.
- ***Dataset***
The Stanford Question Answering Dataset is a question-answering dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator).

    QNLI convert the task into sentence pair classification by forming a pair between each question and each sentence in the corresponding context, and filtering out pairs with low lexical overlap between the question and the context sentence. The task is to determine whether the context sentence contains the answer to the question. 
- ***Usage***
Note: PROJECT_ROOT denotes the project root of this project.

    1. run automatic downloading script
    ```bash
    cd PROJECT_ROOT/dataset
    python get_QNLI.py
    ```
    2. train the model
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/question_nli/conf_qnli_bilstm.json
    ```
     *Tips:can try different model by run different config file.* 
- ***Result***


    | Model    | Accuracy | 
    | -------- | -------- |
    | BiLSTM(GLUE paper)     | 77.0     |
    |BiLSTM+Attn(GLUE paper)|77.2|
    |BiLSTM(NeuronBlocks)|79.80|
    |BiLSTM+Attn(NeuronBlocks)|81.07|
    *Tips: the model file and train log file can find in config file's outputs/save_base_dir after you finish trainning after you finish trainning*

### Task 4: Regression
Regression is another import algorithms in machine learning, and widely used in NLP tasks. The task aim to train a model using labelled dataset containing text documents and their score labels.

Note: PROJECT_ROOT denotes the project root of this project.
- ***Dataset***
We provide a demo dataset in *PROJECT_ROOT/dataset/regression*, you can replace with your own regression dataset for regression task trainning.
- ***Usage***
train regression model
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/regression/conf_regression_bilstm_attn.json
    ```
     *Tips:can train your own model by build different config file.* 

### Task 5: Sentiment Analysis

Sentiment analysis is aimed to predict the sentiment (positive, negative, etc) of a given sentence/document, which is widely applied to many fields.

- ***Dataset***
[The Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) consists of sentences from movie reviews and human annotations of their sentiment. We use the two-way (positive/negative) class split, and use only sentence-level labels.

- ***Usage***
Note: PROJECT_ROOT denotes the project root of this project.

    1. run automatic downloading script
    ```bash
    cd PROJECT_ROOT/dataset
    python get_SST-2.py
    ```
    2. train the model
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/sentiment_analysis/conf_sentiment_analysis_bilstm.json
    ```
     *Tips:can try different model by run different config file.* 
- ***Result***
    
    | Model    | Accuracy |
    | -------- | -------- |
    | BiLSTM (GLUE paper) | 0.875 |
    | BiLSTM+Attn (GLUE paper) | 0.875 |
    | BiLSTM (NeuronBlocks) | 0.876 |
    | BiLSTM+Attn (NeuronBlocks) | 0.883 |
    
    *Tips: the model file and train log file can find in config file's outputs/save_base_dir after you finish trainning*

### Task 6: Question Pairs



- ***Dataset***
[The Quora Question Pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) dataset is a collection of question pairs from the community
question-answering website Quora. The task is to determine whether a pair of questions are semantically equivalent. 

- ***Usage***
Note: PROJECT_ROOT denotes the project root of this project.

    1. run automatic downloading script
    ```bash
    cd PROJECT_ROOT/dataset
    python get_QQP.py
    ```
    2. train the model
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/question_pairs/conf_question_pairs_bilstm.json
    ```
     *Tips:can try different model by run different config file.* 
- ***Result***
The class distribution in QQP is unbalanced (63% negative), so we report both accuracy and F1 score.

    | Model    | Accuracy | F1 |
    | -------- | -------- |-------- |
    | BiLSTM (GLUE paper) | 0.853 | 0.820 |
    | BiLSTM+Attn (GLUE paper) | 0.877 | 0.839 |
    | BiLSTM (NeuronBlocks) | 0.864 | 0.831 |
    | BiLSTM+Attn (NeuronBlocks) | 0.878 | 0.839 |
    
    *Tips: the model file and train log file can find in config file's outputs/save_base_dir*

## Advanced Usage

## FAQ
