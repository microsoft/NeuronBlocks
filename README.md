# ***NeuronBlocks*** - NLP DNN Model Toolkit
* [Overview](#Overview)
* [Who should consider using NeuronBlocks](#Who-should-consider-using-NeuronBlocks)
* [Get Started](#Get-Started)
* [Contribute](#Contribute)
* [Reference](#Reference)
* [Related Project](#Related-Project)
* [License](#License) 
* [Contact](#Contact)



# Overview
NeuronBlocks is a **NLP deep learning modeling toolkit** that helps engineers to build end-to-end pipelines for neural network model training for NLP tasks. The main goal of this toolkit is to minimize developing cost for NLP deep neural network model building, including both training and inferencing stages. For more details, please check our paper.

NeuronBlocks consists of two major components: ***Block Zoo*** and ***Model Zoo***. 
- In ***Block Zoo***, we provide commonly used neural network components as building blocks for model architecture design.  
- In ***Model Zoo***, we provide a suite of NLP models for common NLP tasks, in the form of **JSON configuration** files. 
 
<img src="https://i.imgur.com/LMD0PFQ.png" width="300">

### Toolkit Usage
Users can either pick existing models (config files) in *Model Zoo* to start model training or create new models by leveraging neural network blocks in *Block Zoo* just like playing with Lego. 

<img src="https://i.imgur.com/q0p6Wvz.png" width="300">


### NLP Tasks Supported
- Sentence Classification 
- Question Answering Matching
- Textual Entailment
- Slot tagging
- Machine Reading Comprehension
- *More on-going*

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
- ***Model Visualization***: A model visualizer is provided for visualization and configure correctness checking, which helps users to visualize the model architecture easily during debugging. 
- ***Extensibility***: NeuronBlocks is extensible, allowing users to contribute new blocks or contributing novel models (JSON files).


# Get Started
Please refer to [Tutorial.md](Tutorial.md) and [Code documentation](https://microsoft.github.io/NeuronBlocks/).


# Contribute
NeuronBlocks operates in an open model. It is designed and developed by **STCA NLP Group, Microsoft**. Contributions from academia and industry are also highly welcome.

For more details, please refer to [Contributing.md](Contributing.md).

# Reference
Paper (update soon).

# Related Project
[OpenPAI](https://github.com/Microsoft/pai) is an open source platform that provides complete AI model training and resource management capabilities, it is easy to extend and supports on-premise, cloud and hybrid environments in various scale.

# License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE) License.

# Contact
If you have any questions, please contact NeuronBlocks@microsoft.com

