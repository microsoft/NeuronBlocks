# TMKD
Source model for WSDM 2020 paper "[Model Compression with Two-stage Multi-teacher Knowledge Distillation for Web Question Answering System](https://arxiv.org/abs/1910.08381)". 

## Stage1 model
The link to the pre-trained model:
- [TMKD-Stage1-small-uncased](https://drive.google.com/open?id=1PUr1UOKWpUlsIqAMLnRzXxaOPhfXerVe): 3-layer, 768-hidden,12-heads

## Usage
You can use [Transformers](https://github.com/huggingface/transformers) code repo to load our stage1 model weights directly. 

## Citation
For more information, our work is helpful to you, please kindly cite our paper as follows:
```
@inproceedings{DBLP:conf/wsdm/YangSGLJ20,
  author    = {Ze Yang and
               Linjun Shou and
               Ming Gong and
               Wutao Lin and
               Daxin Jiang},
  editor    = {James Caverlee and
               Xia (Ben) Hu and
               Mounia Lalmas and
               Wei Wang},
  title     = {Model Compression with Two-stage Multi-teacher Knowledge Distillation
               for Web Question Answering System},
  booktitle = {{WSDM} '20: The Thirteenth {ACM} International Conference on Web Search
               and Data Mining, Houston, TX, USA, February 3-7, 2020},
  pages     = {690--698},
  publisher = {{ACM}},
  year      = {2020},
  url       = {https://doi.org/10.1145/3336191.3371792},
  doi       = {10.1145/3336191.3371792},
  timestamp = {Fri, 24 Jan 2020 12:03:51 +0100},
  biburl    = {https://dblp.org/rec/conf/wsdm/YangSGLJ20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

### Reference
* [Transformers](https://github.com/huggingface/transformers)
* [HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/abs/1910.03771)
