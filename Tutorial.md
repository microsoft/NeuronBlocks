# ***NeuronBlocks*** Tutorial

* [Installation](#installation)
* [Quick Start](#quick-start)
* [How to Design Your NLP Model](#design-model)
    * [Define the Model Configuration File](#define-conf)
    * [Visualize Your Model](#visualize)
* [Model Zoo for NLP Tasks](#model-zoo)
    * [Task 1: Text Classification](#task-1)
    * [Task 2: Question Answer Matching](#task-2)
    * [Task 3: Question Natural Language Inference](#task-3)
    * [Task 4: Regression](#task-4)
    * [Task 5: Sentiment Analysis](#task-5)
    * [Task 6: Question Pairs](#task-6)
* [Advanced Usage](#advanced-usage)
    * [Support Extra Feature](#extra-feature)
    * [Learning Rate Decay](#lr-decay)
    * [Fix Embedding Weight & Limit Vocabulary Size](#fix-embedding)
* [Frequently Asked Questions](#faq)

## <span id="installation">Installation</span>

*Note: NeuronBlocks is based on Python 3.6*

1. Clone this project. 
    ```bash
    git clone https://github.com/Microsoft/NeuronBlocks
    ```

2. Install Python packages in requirements.txt by the following command.
    ```bash
    pip install -r requirements.txt
    ```

3. Install PyTorch (*NeuronBlocks supports PyTorch version 0.4.1 currently*).
    
    For **Linux**, run the following command:
    ```bash
    pip install torch==0.4.1
    ```
    
    For **Windows**, we suggest you to install PyTorch via *Conda* by following the instruction of [PyTorch](https://pytorch.org/get-started/previous-versions/).

    

## <span id="quick-start">Quick Start</span>

Get started by trying the given examples.

*Tips: in the following instruction, PROJECTROOT denotes the root directory of this project.*

```bash
# get GloVe pre-trained word vectors
cd PROJECT_ROOT/dataset
bash get_glove.sh

# train
cd PROJECT_ROOT
python train.py --conf_path=model_zoo/demo/conf.json

# test
python test.py --conf_path=model_zoo/demo/conf.json

# predict
python predict.py --conf_path=model_zoo/demo/conf.json
```


## <span id="design-model">How to Design Your NLP Model</span>

### <span id="define-conf">Define the Model Configuration File</span>

To train a neural network, you only need to define your model architecture and some other settings in a JSON configuration file.

You can make your private folder (e.g.*YOURFOLDER*) in *PROJECTROOT/model_zoo/*, then put your model configuration file in *PROJECTROOT/model_zoo/YOURFOLDER/*. In addition, palce the data in *PROJECTROOT/dataset/*.

Take *PROJECTROOT/modelzoo/demo/conf.json* as an example (we make it more suitable for usage explanation so that the model architecture might not be practical), this configuration is used for question answer matching task, which aims to figure out whether the passage can be used as an answer of corresponding quesiton or not. The sample data lies in *PROJECTROOT/dataset/demo/*.

The architecture of the configuration file is:
- **inputs**. This part defines the input configuration.
    - ***use_cache***. If use_cache is true, the toolkit would make cache at the first time so that we can accelerate the training process at the next time.
    - ***dataset_type***. Declare the task type here. Currently, we support classification, regression and so on.
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
    - ***file_header***. [necesssary for training and test] This part defines the file format of train/valid/test data. For instance, the following configuration means there are 3 columns in the data, and we name the first to third columns as question_text, answer_text and label, respectively.
        ```json
        "file_header": {
          "question_text": 0,
          "answer_text": 1,
          "label": 2
        }
        ```
    - ***predict_file_header***. [conditionally necessary for prediction] This part defines the file format of prediction data. **If the file_header of predict_data is not consistent with file_header of train/valid/test data, we have to define "predict_file_header" for predict_data, otherwise conf[inputs][file_header] is applied to the prediction data by default.** Two file_headers are consistent if the indices of data column involved in conf[inputs][model_inputs]) are consistent.
        ```json
        "predict_file_header": {
          "question_text": 0,
          "answer_text": 1
        },
        ```
    - ***file_with_col_header***.[optional, default:False] If your train dataset has column name title, remember to set value to True. Otherwise, it may lead to program error.
    - ***model_inputs***. The node is used for defining model inputs. In this example task, there are two inputs: question and answer:
        ```json
        "model_inputs": {
          "question": [
            "question_text"
          ],
          "answer": [
            "answer_text"
          ]
        }
        ```
    - ***target***. [necessary for training and test, some part necessary for prediction] This node defines the target column in the train/valid/test data.The type of target is array because our tookit support multi-target tasks.
- **outputs**. This node defines the settings of path to save models and logs, as well as cache.
    - ***save_base_dir***. The directory to save models and logs.
    - ***model_name***. The model would be saved as save_base_dir/model_name.
    - ***train_log_name/test_log_name/predict_log_name***. The name of log during training/test/prediction.
    - ***predict_fields***. A list to set up the fields you want to predict, such as prediction and confidence.
    - ***cache_dir***. The directory to save cache.
- **training_params**. We define the optimizer and training hyper parameters here.
    - ***optimizer***. 
        - *name*. We support all the optimizers defined in [torch.optim](http://pytorch.org/docs/0.3.1/optim.html?#module-torch.optim).
        - *params*. The optimizer parameters are exactly the same as the parameters of the initialization function of optimizers in [torch.optim](http://pytorch.org/docs/0.3.1/optim.html?#module-torch.optim).
    - ***use_gpu***. [default: True] Whether to use GPU if there is at least one GPU available. In addition, all resources are used by default if has multi-gpus and you can specify specific gpus use *CUDA_VISIBLE_DEVICES*.
    - ***batch_size***. Define the batch size here. If have multi-gpus, each gpu has same batch as batch_size.
    - ***batch_num_to_show_results***. [necessary for training] During the training process, show the training progress every batch_num_to_show_results batches.
    - ***max_epoch***. [necessary for training] The maximum number of epochs to train.
    - ***valid_times_per_epoch***. [optional for training, default: 1] Define how many times to conduct validation per epoch. Usually, we conduct validation after each epoch, but for a very large corpus, we'd better validate multiple times in case to miss the best state of our model. The default value is 1.
- **architecture**. Define the model architecture. The node is a list to represent layers in block_zoo of a model. The supported layers of this tool are given on [Block_zoo overview](http://10.177.74.200:8080/layers.html). 
    
    - ***Embedding layer***. The first layer of this example (as shown below) defines the embedding layer, which is composed of two types of embedding: "word" (word embedding). The name of embedding types can be defined as any other name. It denotes that the the columns related to "word" are question_text and answer_text. Lastly, the dimension of "word" are 300.
        ```json
        {
            "layer": "Embedding",
            "conf": {
              "word": {
                "cols": ["question_text", "answer_text"],
                "dim": 300
              }
             }
        }
        ```
    - Using block_zoo to design your model. You can choose layers in block_zoo to build your model follow the fomat:
        - *layer_id*.Customized name for one model layer
        - *layer*. The layer name in block_zoo  
        - *conf*. Each layer has their own config(you can find layer name and corresponding parameters in [Block_zoo overview](http://10.177.74.200:8080/layers.html))
        - *inputs*. The layer_id which connect to this layer, the type of inputs must be array, because one layer can have multi-layer input.
    This is a BiLSTM layer example: 
        ```json
        {
            "layer_id": "question_1",
            "layer": "BiLSTM",
            "conf": {
              "hidden_dim": 64,
              "dropout": 0
            },
            "inputs": ["question"]
        }
        ```
        To know more about supported layers and their configuration, please go to [Block_zoo overview](http://10.177.74.200:8080/layers.html). For example, if we want to know the parameters of BiLSTM, we can find that there is a BiLSTM class and a BiLSTMConf class, the parameters of BiLSTM would be given at BiLSTMConf.
- **loss**. [necessary for training and test] Currently, we support all the loss functions offered by [PyTorch loss functions](http://pytorch.org/docs/0.3.1/nn.html#loss-functions). The parameters defined in configuration/loss/conf are exactly the same with the parameters of initialization function of loss functions in [PyTorch loss functions](http://pytorch.org/docs/0.3.1/nn.html#loss-functions). Additionally, we offer more options, such as [Focal Loss](https://arxiv.org/pdf/1708.02002.pdf), please refer to [Loss function overview](http://10.177.74.200:8080/losses.html).
        Specially, for classification tasks, we usually add a Linear layer to project the output to dimension of number of classes, if we don't know the #classes, we can use '-1' instead and we would calculate the number of classes from the corpus.
        
- **metrics**. Different tasks have different supported metrics, you can follow the table below to select metrics according specific task.


     Task | Supported Metrics 
     -------- | --------  
     classification     | auc, accuracy, f1, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall, weighted_f1, weighted_precision, weighted_recall     
     sequence_tagging|seq_tag_f1, accuracy
     regression |MSE, RMSE
     mrc | F1, EM
    
    During validation, the toolkit select the best model according to the first metric.

*Tips: the [optional] and [necessary] mark means corresponding node in the configuration file is optional or necessary for training/test/prediction. If there is no mark, it means the node is necessary all the time. Actually, it would be more convenient to prepare a configuration file that contains all the configurations for training, test and prediction.*



## <span id="visualize">Visualize Your Model</span>

A model visualizer is provided for visualization and configuration correctness checking, please refer to [MV_README.md](MV_README.md).

## <span id="model-zoo">Model Zoo for NLP Tasks</span>

In Model Zoo, we provide a suite of NLP models for common NLP tasks, in the form of JSON configuration files. You can pick one of existing models (JSON config files) in Model Zoo to start model training quickly, or build your own models by modifying the JSON config file to suit your specific task.


### <span id="task-1">Task 1: Text Classification</span>

Text classification is a core problem to many applications like spam filtering, email routing, book classification, etc. This task aim to train a classifier using labelled dataset containing text documents and their labels, which can be web pages, papers, emails etc.

- ***Dataset***

    The [20 Newsgroups data set](http://qwone.com/~jason/20Newsgroups/) is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. Here is a list of the 20 newsgroups, partitioned (more or less) according to subject matter:

    ![](https://i.imgur.com/rHLITSi.png)

- ***Usage***

    1. run data downloading and preprocessing script.
    ```bash
    cd PROJECT_ROOT/dataset
    python get_20_newsgroups.py
    ```
    2. train text classification model.
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/text_classification/conf_text_classification_cnn.json 
    ```
     *Tips: you can try different models by running different JSON config files.*

- ***Result***
    
     Model    | Accuracy 
     -------- | -------- 
     TextCNN  | 0.961 
     BiLSTM+Attention | 0.970 
    
    *Tips: the model file and train log file can be found in JOSN config file's outputs/save_base_dir after you finish training.*


### <span id="task-2">Task 2: Question Answer Matching</span>

Question answer matching is a crucial subtask of the question answering problem, with the aim of determining whether question-answer pairs are matched or not.

- ***Dataset***

    [Microsoft Research WikiQA Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52419) is a publicly available set of question and sentence pairs, collected and annotated for research on open-domain question answer matching. WikiQA includes 3,047 questions and 29,258 sentences, where 1,473 sentences were labeled as answer sentences to their corresponding questions. More detail of this corpus can be found in the paper [WikiQA: A Challenge Dataset for Open-Domain Question Answering](https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/).

- ***Usage***

    1. run data downloading script.
    ```bash
    cd PROJECT_ROOT/dataset
    python get_WikiQACorpus.py
    ```
    
    2. train question answer matching model.
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/question_answer_matching/conf_question_answer_matching_bilstm.json
    ```
    
     *Tips: you can try different models by running different JSON config files.*
     
- ***Result***
    
     Model    | AUC 
     -------- | -------- 
     CNN (WikiQA paper) | 0.735 
     CNN-Cnt (WikiQA paper) | 0.753 
     CNN (NeuronBlocks) | 0.747 
     BiLSTM (NeuronBlocks) | 0.767 
     BiLSTM+Attn (NeuronBlocks) | 0.754 
    
    *Tips: the model file and train log file can be found in JOSN config file's outputs/save_base_dir after you finish training.*

### <span id="task-3">Task 3: Question Natural Language Inference</span>

Natural language inference (NLI) is a task that incorporates much of what is necessary to understand language, such as the ability to leverage world knowledge or perform lexico-syntactic reasoning. Given two sentences, a premise and a hypothesis, an NLI system must determine whether the hypothesis is implied by the premise.

- ***Dataset***

    The Stanford Question Answering Dataset is a question-answering dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator).
    
    QNLI convert the task into sentence pair classification by forming a pair between each question and each sentence in the corresponding context, and filtering out pairs with low lexical overlap between the question and the context sentence. The task is to determine whether the context sentence contains the answer to the question. 

- ***Usage***

    1. run data downloading script.
    ```bash
    cd PROJECT_ROOT/dataset
    python get_QNLI.py
    ```
    2. train the model.
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/question_nli/conf_qnli_bilstm.json
    ```
     *Tips: you can try different models by running different JSON config files.*

- ***Result***


     Model    | Accuracy  
     -------- | -------- 
     BiLSTM(GLUE paper)     | 0.770 
     BiLSTM+Attn(GLUE paper) | 0.772 
     BiLSTM(NeuronBlocks) | 0.798 
     BiLSTM+Attn(NeuronBlocks) | 0.810 
    
    *Tips: the model file and train log file can be found in JOSN config file's outputs/save_base_dir after you finish training.*

### <span id="task-4">Task 4: Regression</span>

Regression is the problem of predicting a continuous number for given input, widely used in NLP tasks. This task aims to train a model using dataset containing text documents and their scores.

- ***Dataset***

    We provide a demo dataset in *PROJECT_ROOT/dataset/regression*, you can replace with your own regression dataset for regression task trainning.

- ***Usage***

    Train regression model.
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/regression/conf_regression_bilstm_attn.json
    ```
     *Tips: you can try different models by running different JSON config files.*

### <span id="task-5">Task 5: Sentiment Analysis</span>

Sentiment analysis is aimed to predict the sentiment (positive, negative, etc) of a given sentence/document, which is widely applied to many fields.

- ***Dataset***

    [The Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) consists of sentences from movie reviews and human annotations of their sentiment. We use the two-way (positive/negative) class split, and use only sentence-level labels.

- ***Usage***

    1. run data downloading script.
    ```bash
    cd PROJECT_ROOT/dataset
    python get_SST-2.py
    ```
    2. train the model.
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/sentiment_analysis/conf_sentiment_analysis_bilstm.json
    ```
     *Tips: you can try different models by running different JSON config files.*
     
- ***Result***
    
     Model    | Accuracy 
     -------- | -------- 
     BiLSTM (GLUE paper) | 0.875 
     BiLSTM+Attn (GLUE paper) | 0.875 
     BiLSTM (NeuronBlocks) | 0.876 
     BiLSTM+Attn (NeuronBlocks) | 0.883 
    
    *Tips: the model file and train log file can be found in JOSN config file's outputs/save_base_dir after you finish training.*

### <span id="task-6">Task 6: Question Pairs</span>

This task is to determine whether a pair of questions are semantically equivalent. 

- ***Dataset***

    [The Quora Question Pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) dataset is a collection of question pairs from the community question-answering website Quora.

- ***Usage***

    1. run data downloading script.
    ```bash
    cd PROJECT_ROOT/dataset
    python get_QQP.py
    ```
    2. train the model.
    ```bash
    cd PROJECT_ROOT
    python train.py --conf_path=model_zoo/nlp_tasks/question_pairs/conf_question_pairs_bilstm.json
    ```
     *Tips: you can try different models by running different JSON config files.*

- ***Result***
    The class distribution in QQP is unbalanced (63% negative), so we report both accuracy and F1 score.
    
     Model    | Accuracy | F1 
     -------- | -------- |-------- 
     BiLSTM (GLUE paper) | 0.853 | 0.820 
     BiLSTM+Attn (GLUE paper) | 0.877 | 0.839 
     BiLSTM (NeuronBlocks) | 0.864 | 0.831 
     BiLSTM+Attn (NeuronBlocks) | 0.878 | 0.839 
    
    *Tips: the model file and train log file can be found in JSON config file's outputs/save_base_dir.*

## <span id="advanced-usage">Advanced Usage</span>

After building a model, the next goal is to train a good performance model. It depends on a highly expressive model and tricks of the model training. NeuronBlocks provides most common tricks of model training.

Take *PROJECTROOT/modelzoo/advanced/conf.json* as an example (we make it more suitable for the usage explanation so that the model architecture might not be practical) to introduce the advanced usage, the configuration is used for question answer matching task. The sample data lies in *PROJECTROOT/dataset/advanced/demo*.

### <span id="extra-feature">Support Extra Feature</span>

Providing more features (postag, NER, char-level feature, etc) to the model than just a single original text may obtain more improvement in performance. NeuronBlocks supports multi-feature input and embedding.

To achieve it, you need:

  1. Specify the corresponding column name in config file's inputs/file_header
  *NOTE: char-level feature doesn't need to be specified*
  ```bash
      "file_header": {
          "question_text": 0,
          "answer_text": 1,
          "label":   2,
          "question_postag": 3,
          "answer_postag": 4
        }
  ```
  2. Specify the corresponding feature name in config file's inputs/model_inputs
  ```bash
      "model_inputs": {
          "question": ["question_text","question_postag","question_char"],
          "answer": ["answer_text","answer_postag","answer_char"]
        }
  ```
  3. Set the corresponding feature embedding in config file's architecture Embedding layer
  ```bash
      {
            "layer": "Embedding",
            "conf": {
              "word": {
                "cols": ["question_text", "answer_text"],
                "dim": 300,
                "fix_weight": true
              },
              "postag": {
                "cols": ["question_postag","answer_postag"],
                "dim": 20
              },
              "char": {
                "cols": ["question_char", "answer_char"],
                "type": "CNNCharEmbedding",
                "dropout": 0.2,
                "dim": 30,
                "embedding_matrix_dim": 8,
                "stride":1,
                "window_size": 5,
                "activation": null
              }
            }
        }
  ```
### <span id="lr-decay">Learning Rate Decay</span>

The learning rate is one of the most important hyperparameters to tune during training. Choosing the learning rate is challenging. A too small value may result in a long training process that could get stuck, while a too large value may result in learning a sub-optimal set of weights too fast or an unstable training process.

When training a model, it is often recommended to lower the learning rate as the training progresses. NeuronBlocks provides corresponding function for supporting learning_rate decay by setting several parameters in config files.

***training_params/lr_decay***. [float, optional for training] lr_decay is a float number between 0 and 1, and learning rate will multiply by it at each epoch after *epoch_start_lr_decay*.

***training_params/minimum_lr***. [float, optional for training] The minimum learning rate during training. Once less than it, the learning rate will be replaced by minimum_lr.

***training_params/epoch_start_lr_decay***. [int, optional for training] The epoch number of starting decay learning rate.

Learing rate decay example:
```bash
"optimizer": {
  "name": "Adam",
  "params": {
    "lr": 0.001
  }
},
"lr_decay": 0.95,
"minimum_lr": 0.0001,
"epoch_start_lr_decay": 1
```

### <span id="fix-embedding">Fix Embedding Weight & Limit Vocabulary Size</span>

When corpus is very large, the vocabulary size will become large correspondingly. Moreover the training process will be slow if the vocabulary embedding vectors keep updating during training. 

To solve the above problems, NeuronBlocks supports *fix embedding weight (embedding vectors don't update during training)* and *limit vocabulary size*.

- **Fix embedding weight**
    
    ***fix_weight***. [bool, optional for training, default: false] By setting *fix_weight* parameter in architecture/Embedding layer, you can control the embeding vectors is updatable or not during training.
    
    For example, setting word embedding not updatable:
    ```bash
    {
      "layer": "Embedding",
      "conf": {
        "word": {
          "cols": ["question_text", "answer_text"],
          "dim": 300,
          "fix_weight": true
        }
      }
    }
    ```
    
- **Limit vocabulary size**
    
    ***training_params/vocabulary/min_word_frequency***. [int, optional for training, default: 3] The word will be removed in corpus vocabulary if its statistical frequency less than min_word_frequency.
    
    ***training_params/vocabulary/max_vocabulary***. [int, optional for training, default: 800,000] The max size number of corpus vocabulary. If corpus vocabulary size is larger than max_vocabulary, it will be cut according to word frequency.

## <span id="faq">Frequently Asked Questions</span>

