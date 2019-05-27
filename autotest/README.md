# ***NeuronBlocks*** AUTOTEST

1. Please download GloVe firstly via following commands.
```bash
cd PROJECT_ROOT/dataset
# windows
./get_glove.sh
# linux
sh get_glove.sh
```
2. Please download word vectors from [Chinese Word Vectors (SGNS/Mixed-large)](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw) and bunzip , 
then place it in a directory (e.g. PROJECT_ROOT/dataset/chinese_word_vectors/). 
3. Please download the [20 Newsgroups data set](http://qwone.com/~jason/20Newsgroups/). You can run the following data downloading and preprocessing script.
```bash
cd PROJECT_ROOT/dataset
python get_20_newsgroups.py
```
4. Please run autotest script.
```bash
# windows
./ autotest.sh A B
# linux
sh autotest.sh A B
```
where, parameter A indicates single process or multiple processes, the default is single process. When A is Y, it stands for multiple processes. 
Parameter B indicates using GPU or CPU to test, the default is using CPU. When B is not empty, you need to specify which GPUs to use.
```bash
# Using multiple processes with the first GPU
sh autotest.sh Y 1
# Using multiple processes with CPU
sh autotest.sh Y
# Using single processes with the first GPU
sh autotest.sh N 1
# Using single processes with CPU
sh autotest.sh N
```
5. Finally, you can get the contrast_results.txt in PROJECT_ROOT/autotest, which stores the results of your model.
