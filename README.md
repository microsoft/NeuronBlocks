# ***NeuronBlocks*** - Building Your NLP DNN Models Like Playing Lego

![language](https://img.shields.io/badge/language-en%7C中文-brightgreen.svg)
[![python](https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/pytorch-0.4%20%7C%201.x-orange.svg)](https://pytorch.org)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

[简体中文](README_zh_CN.md)


# Table of Contents
* [Overview](#Overview)
* [Get Started in 60 Seconds](#Get-Started-in-60-Seconds)
* [Who should consider using NeuronBlocks](#Who-should-consider-using-NeuronBlocks)
* [Contribute](#Contribute)
* [Reference](#Reference)
* [Related Project](#Related-Project)
* [License](#License) 
* [Contact](#Contact)



# Overview
NeuronBlocks is a **NLP deep learning modeling toolkit** that helps engineers/researchers to build end-to-end pipelines for neural network model training for NLP tasks. The main goal of this toolkit is to minimize developing cost for NLP deep neural network model building, including both training and inference stages.

NeuronBlocks consists of two major components: ***Block Zoo*** and ***Model Zoo***. 
- In ***Block Zoo***, we provide commonly used neural network components as building blocks for model architecture design.  
- In ***Model Zoo***, we provide a suite of NLP models for common NLP tasks, in the form of **JSON configuration** files. 
 
<img src="https://i.imgur.com/LMD0PFQ.png" width="300">

### NLP Tasks Supported
- Sentence Classification 
- Sentiment Analysis 
- Question Answering Matching
- Textual Entailment
- Slot tagging
- Machine Reading Comprehension
- Knowledge Distillation for Model Compression
- *More on-going*

### Toolkit Usage
Users can either pick existing models (config files) in *Model Zoo* to start model training or create new models by leveraging neural network blocks in *Block Zoo* just like playing with Lego. 

<img src="https://i.imgur.com/q0p6Wvz.png" width="300">


# Get Started in 60 Seconds
## <span id="installation">Installation</span>

*Note: NeuronBlocks requires **Python 3.6*** and above.

1. Clone this project. 
    ```bash
    git clone https://github.com/Microsoft/NeuronBlocks
    ```

2. Install Python packages in requirements.txt by the following command.
    ```bash
    pip install -r requirements.txt
    ```

3. Install PyTorch (*NeuronBlocks supports **PyTorch 0.4.1** and above*).
    
    For **Linux**, run the following command:
    ```bash
    pip install "torch>=0.4.1"
    ```
    
    For **Windows**, we suggest you to install PyTorch via *Conda* by following the instruction of [PyTorch](https://pytorch.org/get-started/locally/).
    

## <span id="quick-start">Quick Start</span>

Get started by trying the given examples. Both **Linux/Windows, GPU/CPU** are supported. For **Windows**, we suggest you to use PowerShell instead of CMD.

*Tips: in the following instruction, PROJECTROOT denotes the root directory of this project.*

```bash
# train
cd PROJECT_ROOT
python train.py --conf_path=model_zoo/demo/conf.json

# test
python test.py --conf_path=model_zoo/demo/conf.json

# predict
python predict.py --conf_path=model_zoo/demo/conf.json
```
For more details, please refer to [Tutorial.md](Tutorial.md) and [Code documentation](https://microsoft.github.io/NeuronBlocks/).

# Who should consider using NeuronBlocks
Engineers or researchers who face the following challenges when using neural network models to address NLP problems: 
+ Many frameworks to choose and high framework studying cost. 
+ Heavy coding cost. A lot of details make it hard to debug.
+ Fast Model Architecture Evolution. It is difficult for engineers to understand the mathematical principles behind them.
+ Model Code optimization requires deep expertise.
+ Model Platform Compatibility Requirement.  It requires extra coding work for the model to run on different platforms, such as Linux/Windows, GPU/CPU. 


The advantages of leveraging NeuronBlocks for NLP neural network model training includes:
- ***Model Building***: for model building and parameter tuning, users only need to write simple JSON config files, which greatly minimize the effort of implementing new ideas.
- ***Model Sharing*** It is super easy to share models just through JSON files, instead of nasty codes. For different models or tasks, our users only need to maintain one single centralized source code base.
- ***Code Reusability***: Common blocks can be easily shared across various models or tasks, reducing duplicate coding work.  
- ***Platform Flexibility***: NeuronBlocks can run both on Linux and Windows machines, using both CPU and GPU. It also supports training on GPU platforms like Philly and PAI.

    <table align="center">
        <tr><td align="center"></td><td align="center">CPU inference</td><td align="center">Single-GPU inference</td><td align="center">Multi-GPU inference</td></tr>
        <tr><td align="center">CPU train</td><td align="center">&#10003;</td><td align="center">&#10003;</td><td align="center">&#10003;</td></tr>
        <tr><td align="center">Single-GPU train</td><td align="center">&#10003;</td><td align="center">&#10003;</td><td align="center">&#10003;</td></tr>
        <tr><td align="center">Multi-GPU train</td><td align="center">&#10003;</td><td align="center">&#10003;</td><td align="center">&#10003;</td></tr>
    </table>
- ***Model Visualization***: A model visualizer is provided for visualization and configure correctness checking, which helps users to visualize the model architecture easily during debugging. 
- ***Extensibility***: NeuronBlocks is extensible, allowing users to contribute new blocks or contributing novel models (JSON files).

# Contribute
NeuronBlocks operates in an open model. It is designed and developed by **STCA NLP Group, Microsoft**. Contributions from academia and industry are also highly welcome. For more details, please refer to [Contributing.md](Contributing.md).

## Ongoing Work and Call for Contributions
Anyone who are familiar with are highly encouraged to contribute code.
* Knowledge Distillation for Model Compression. Knowledge distillation for heavy models such as BERT, OpenAI Transformer. Teacher-Student based knowledge distillation is one common method for model compression. 
* Multi-Lingual Support
* NER Model Support 
* Multi-Task Training Support 

# Reference
**NeuronBlocks -- Building Your NLP DNN Models Like Playing Lego**, at https://arxiv.org/abs/1904.09535.

# Related Project
* [OpenPAI](https://github.com/Microsoft/pai) is an open source platform that provides complete AI model training and resource management capabilities, it is easy to extend and supports on-premise, cloud and hybrid environments in various scale.
* [Samples for AI](https://github.com/Microsoft/samples-for-ai):  a deep learning samples and projects collection. It contains a lot of classic deep learning algorithms and applications with different frameworks, which is a good entry for the beginners to get started with deep learning.

# License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE) License.

# Contact
If you have any questions, please contact NeuronBlocks@microsoft.com

If you have wechat, you can also add the following account:

<img src="https://i.imgur.com/lI2oQWo.jpg" width="200">

