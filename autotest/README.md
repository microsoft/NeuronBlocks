# ***NeuronBlocks*** AUTOTEST

1. Please download GloVe firstly via following commands.
```bash
cd PROJECT_ROOT/dataset
# windows
./get_glove.sh
# linux
sh get_glove.sh
```
2. Please download the [20 Newsgroups data set](http://qwone.com/~jason/20Newsgroups/). You can run the following data downloading and preprocessing script.
```bash
cd PROJECT_ROOT/dataset
python get_20_newsgroups.py
```
3. Please run the autotest script.
```bash
# windows
./autotest.sh A B
# linux
sh autotest.sh A B
```
where, parameter A indicates single process or multiple processes, the default is single process. When A is Y, it stands for multiple processes. 
Parameter B indicates using GPU or CPU to test, the default is using CPU. When B is not empty, you need to specify which GPUs to use.
You can choose any one of the following scripts according to your needs.
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
4. Finally, you can get the contrast_results.txt in PROJECT_ROOT/autotest, which stores the results of your model.
You can compare the results of column {accuracy/new AUC} versus column {old accuracy/AUC}. If there are significant metric regression, you need to check your pull request. 

```
tasks                   GPU/CPU old accuracy/AUC    accuracy/new AUC 
english_text_matching       GPU     0.96655         0.97375
english_text_matching       CPU     0.96655         0.97375
chinese_text_matching       GPU     0.70001         0.7 
chinese_text_matching       CPU     0.70001         0.7 
quora_question_pairs        GPU     0.72596         0.727864
quora_question_pairs        CPU     0.72596         0.727864
knowledge_distillation      CPU     0.66329         0.6695541666666667
```
