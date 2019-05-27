# ***NeuronBlocks*** AUTOTEST

1. Please download GloVe firstly via following commands.
```bash
cd PROJECT_ROOT/dataset
./get_glove.sh
```
2. Please download word vectors from [Chinese Word Vectors](https://github.com/Embedding/Chinese-Word-Vectors#pre-trained-chinese-word-vectors) and bunzip , 
then place it in a directory (e.g. PROJECT_ROOT/dataset/chinese_word_vectors/). Then remember to define inputs/data_paths/pre_trained_emb in 
PROJECT_ROOT/autotest/conf/conf_chinese_text_matching_emb_char_autotest.json.
3. Please download the [20 Newsgroups data set](http://qwone.com/~jason/20Newsgroups/). You can run the following data downloading and preprocessing script.
```bash
cd PROJECT_ROOT/dataset
python get_20_newsgroups.py
```
4. Please run autotest script.
```bash
sh autotest.sh A B
```
where, parameter A indicates single process or multiple processes, the default is single process. When A is Y, it stands for multiple processes. 
Parameter B indicates using GPU or CPU to test, the default is using CPU. When B is not empty, you need to specify which graphics card to use.
For example, you run the following script indicating that you use multiple processes with the first graphics card.
```bash
sh autotest.sh Y 1
```
5. Finally, you can get the contrast_results.txt in PROJECT_ROOT/autotest, which stores the effect of your model.
