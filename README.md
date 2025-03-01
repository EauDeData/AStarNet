# A\*Net: A\* Networks #

This is the official codebase of the paper

[A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs][paper]

[Zhaocheng Zhu](https://kiddozhu.github.io)\*,
[Xinyu Yuan](https://github.com/KatarinaYuan)\*,
[Mikhail Galkin](https://migalkin.github.io),
[Sophie Xhonneux](https://github.com/lpxhonneux),
[Ming Zhang](http://net.pku.edu.cn/dlib/mzhang/),
[Maxime Gazeau](https://scholar.google.com/citations?user=LfmqBJsAAAAJ),
[Jian Tang](https://jian-tang.com)

[paper]: https://arxiv.org/pdf/2206.04798.pdf

## Overview ##

A\*Net is a scalable path-based method for knowledge graph reasoning. Inspired by
the classical A\* algorithm, A\*Net learns a neural priority function to select
important nodes and edges at each iteration, which significantly reduces time and
memory footprint for both training and inference.

A\*Net is the first path-based method that scales to ogbl-wikikg2 (2.5M entities,
16M triplets). It also enjoys the advantages of path-based methods such as
inductive capacity and interpretability.

Here is a demo of A\*Net with a ChatGPT interface. By reasoning on the Wikidata
knowledge graph, ChatGPT produces more grounded predictions and less hallucination.

![A*Net with ChatGPT interface](asset/chat.png)

https://github.com/DeepGraphLearning/AStarNet/assets/17213634/b521113e-1360-4082-af65-e2579bf01b29

This codebase contains implementation for A\*Net and its predecessor [NBFNet].

[NBFNet]: https://github.com/DeepGraphLearning/NBFNet

## Installation ##

I have ran through ashes, dust and bones to make this installation work.
find here a short script for minimal instalation of a working environment in CUDA 12.4 and PyTorch 2.40:

```bash
conda create -n astar python==3.10 -y
conda activate astar; conda clean -a -y; pip cache purge
pip install numpy==1.26.4
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
echo "Now go and thank this brother git@github.com:anilakash/torchdrug.git"
git clone git@github.com:anilakash/torchdrug.git
cd torchdrug
pip install -r requirements.txt
python setup.py install
cd ..; rm -rf ./torchdrug

pip install ogb
pip install easydict
pip install PyYAML
pip install easydict
```

## Usage ##

To run A\*Net, use the following command. The argument `-c` specifies the experiment
configuration file, which includes the dataset, model architecture, and
hyperparameters. You can find all configuration files in `config/.../*.yaml`.
All the datasets will be automatically downloaded in the code.

```bash
python script/run.py -c config/transductive/fb15k237_astarnet.yaml --gpus [0]
```

For each experiment, you can specify the number of GPU via the argument `--gpus`.
You may use `--gpus null` to run A\*Net on a CPU, though it would be very slow.
To run A\*Net with multiple GPUs, launch the experiment with `torchrun`

```bash
torchrun --nproc_per_node=4 script/run.py -c config/transductive/fb15k237_astarnet.yaml --gpus [0,1,2,3]
```

For the inductive setting, there are 4 different splits for each dataset. You need
to additionally specify the split version with `--version v1`.

## ChatGPT Interface ##

We provide a ChatGPT interface of A\*Net, where users can interact with A\*Net
through natural language. To play with the ChatGPT interface, download the
checkpoint [here] and run the following command. Note you need an OpenAI API key
to run the demo.

```bash
export OPENAI_API_KEY=your-openai-api-key
python script/chat.py -c config/transductive/wikikg2_astarnet_visualize.yaml --checkpoint wikikg2_astarnet.pth --gpus [0]
```

[here]: https://drive.google.com/drive/folders/15NtyKEXnP4NkHIZEArfTE04Tn5PjpbpJ?usp=sharing

## Visualization ##

A\*Net supports visualization of important paths for its predictions. With a trained
model, you can visualize the important paths with the following line. Please replace
the checkpoint with your own path.

```bash
python script/visualize.py -c config/transductive/fb15k237_astarnet_visualize.yaml --checkpoint /path/to/astarnet/experiment/model_epoch_20.pth --gpus [0]
```

## Parameterize with your favourite GNNs ##

A\*Net is designed to be general frameworks for knowledge graph reasoning. This
means you can parameterize it with a broad range of message-passing GNNs. To do so,
just implement a convolution layer in `reasoning/layer.py` and register it with
`@R.register`. The GNN layer is expected to have the following member functions

```python
def message(self, graph, input):
    ...
    return message

def aggregate(self, graph, message):
    ...
    return update

def combine(self, input, update):
    ...
    return output
```

where the arguments and the return values are
- `graph` ([data.PackedGraph]): a batch of subgraphs selected by A*Net, with
  `graph.query` being the query embeddings of shape `(batch_size, input_dim)`.
- `input` (Tensor): node representations of shape `(graph.num_node, input_dim)`.
- `message` (Tensor): messages of shape `(graph.num_edge, input_dim)`.
- `update` (Tensor): aggregated messages of shape `(graph.num_node, *)`.
- `output` (Tensor): output representations of shape `(graph.num_node, output_dim)`.

To support the neural priority function in A\*Net, we need to additionally provide
an interface for computing messages

```python
def compute_message(self, node_input, edge_input):
   ...
   return msg_output
```

You may refer to the following tutorials of TorchDrug
- [Graph Data Structures](https://torchdrug.ai/docs/notes/graph.html)
- [Graph Neural Network Layers](https://torchdrug.ai/docs/notes/layer.html)

[data.PackedGraph]: https://torchdrug.ai/docs/api/data.html#packedgraph

## Frequently Asked Questions ##

1. **The code is stuck at the beginning of epoch 0.**

   This is probably because the JIT cache is broken.
   Try `rm -r ~/.cache/torch_extensions/*` and run the code again.

## Citation ##

If you find this project useful, please consider citing the following paper

```bibtex
@article{zhu2022scalable,
  title={A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs},
  author={Zhu, Zhaocheng and Yuan, Xinyu and Galkin, Mikhail and Xhonneux, Sophie and Zhang, Ming and Gazeau, Maxime and Tang, Jian},
  journal={arXiv preprint arXiv:2206.04798},
  year={2022}
}
```
