# ***NeuronBlocks*** Model Visualizer

In ***NeuronBlocks***, a model visualizer is provided for visualization and configuration correctness checking, 
which helps users to visualize the model architecture easily during debugging.

## Installation
Two libraries are needed for ***NeuronBlocks*** Model Visualizer: graphviz, web.py.

You can install them via pip:

```bash
pip install graphviz
pip install web.py==0.40.dev0
```

*Note: For Windows, please visit [Graphviz Windows Packages](https://graphviz.gitlab.io/_pages/Download/Download_windows.html) to 
download graphviz, then install it and set the PATH variable.*

## Usage

Model Visualizer has 2 modes:
 - **Command Line Mode**: View model architecture via command line.
 - **Browser Mode**: Firstly build a model visualizer server, then view model architecture via a browser.

### Command Line Mode

View model architecture via command line mode, by running:
```bash
python get_model_graph.py --conf_path ../model_zoo/demo/conf.json --graph_path ./graph
```
Arguments:
```bash
--conf_path: [necessary] Path of the input JSON config file.
--graph_path: [optional, default: './graph'] Path of the ouput model graph file.
```
You will get two file: *graph.gv* and *graph.gv.svg*.
Open *graph.gv.svg*, then view the model architecture.

### Browser Mode

Firstly, start Model Visualizer server:

```bash
cd server/
python main.py 8080
```
Then, you can access a model visualizer in your browser.
For Linux, visit http://<your_machine_ip>:8080 .
For Windows, visit http://localhost:8080 .

Finally, input the JSON config in *Config Json* field, click *Submit* button, 
and get the model architecture.
