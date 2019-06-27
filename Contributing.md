
# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


# How to Contribute
- Contribute Model to **Model Zoo**
    - We encourage everyone to contribute their NLP models (namely JSON configuration files). Please follow the structure in model_zoo to  create a pull request.
- Contribute Block to **Block Zoo**

    We encourage everyone to improve this toolkit by contributing code, such as customized Blocks. So other users can further benefit from these new Blocks.
    
    For adding a new block to NeuronBlocks, you need *Three steps*(take [BiLSTM block](https://github.com/microsoft/NeuronBlocks/blob/master/block_zoo/BiLSTM.py) for example):
    - Define the new block's Configuration class(BiLSTMConf class in BiLSTM block). The Configuration class should inheritance [Base Configuration Class](https://github.com/microsoft/NeuronBlocks/blob/master/block_zoo/BaseLayer.py) that define some necessary functions, and rewrite these functions.
    We will give the details of these functions:
        ```bash
        def default():
            '''
            Define the default hyper parameters for block, it will read the corresponding block hyper parameters in configuration json files firstly.
            '''
        
        def declare():
            '''
            Define things like "input_ranks" and "num_of_inputs", which are certain with regard to the block.
            num_of_input is N(N>0) means this layer accepts N inputs;
            num_of_input is -1 means this layer accepts any number of inputs;
            
            The rank here is not the same as matrix rank:
                For a scalar, its rank is 0;
                For a vector, its rank is 1;
                For a matrix, its rank is 2;
                For a cube of numbers, its rank is 3.
            
            if num_of_input > 0:
                len(input_ranks) should be equal to num_of_input
            elif num_of_input == -1:
                input_ranks should be a list with only one element and the rank of all the inputs should be equal to that element.
            '''
        
        def inference():
            '''
            Inference things like output_dim, which may relies on defined hyper parameter or the block special operation.
            '''
        
        def verify():
            '''
            Define some necessary varification for your layer when we define the model.
            '''
        ```
    - Implement the new block's class(BiLSTM class in BiLSTM block). The block class should inheritance [Base Block Class](https://github.com/microsoft/NeuronBlocks/blob/master/block_zoo/BaseLayer.py) and rewrite __init__ and forward function.
        ```bash
        def __init__():
            '''
            Define necessary attributions that would be used in block operation logic.
            '''
        
        def forward():
            '''
            Tensor operation logic.
            '''
        ```
    - Register the new block in block_zoo.
        NeuronBlocks provides a script that can register new block automatically, and blocks contributors just focus on block logic.        
        *Tips: PROJECTROOT denotes the root directory of this project.*
        ```bash
        cd PROJECT_ROOT
        python register_Block.py --block_name=new_block_name
        ```
    
*Tips: Before you contribute your code, we strongly suggest to verify that your improvements are valid by **[AUTOTEST](./autotest)**. We also encourage everyone to improve this autotest tool by contributing code, such as adding test tasks.*
