# BLIVA: A Simple Multimodal LLM for Better Handling of Text-rich Visual Questions
[Wenbo Hu*](https://gordonhu608.github.io/), [Yifan Xu*](https://yfxu.com/), [Yi Li](https://jerryli1019.github.io/), [Weiyue Li](https://weiyueli7.github.io/), [Zeyuan Chen](https://zeyuan-chen.com/), and [Zhuowen Tu](https://pages.ucsd.edu/~ztu/). *Equal Contribution

**UC San Diego**, **Coinbase Global, Inc.**

<a href='https://gordonhu608.github.io/bliva/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs/2308.09936'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/spaces/mlpc-lab/BLIVA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> <a href='https://huggingface.co/mlpc-lab/BLIVA_Vicuna'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-BLIVA_Vicuna_Model-blue'></a> <a href='https://huggingface.co/mlpc-lab/BLIVA_FlanT5'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-BLIVA_FlanT5_Model-blue'></a> <a href='https://huggingface.co/datasets/mlpc-lab/YTTB-VQA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Youtube Thumbnails Datasets-blue'></a>

<p align="center">
    <a href="https://huggingface.co/spaces/mlpc-lab/BLIVA"><img src="images/arch.png" width="90%"></a> <br> Our model architecture in detail with example responses.
</p>

## Release
- [8/21] 🔥 We released **BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions**.  Checkout the [paper](https://arxiv.org/abs/2308.09936). 
- [8/21] 🔥 We released our demo at [here](https://huggingface.co/spaces/mlpc-lab/BLIVA) which is publicly available for everyone to play with. 
- [8/21] We released our Model weight for BLIVA Vicuna Version at [here](https://huggingface.co/mlpc-lab/BLIVA_Vicuna) and FLAN T5 Version at [here](https://huggingface.co/mlpc-lab/BLIVA_FlanT5) which is available for commercial use. 
- [8/21] Our Youtube Visual Question Answering Dataset (YTTB-VQA) is available at [here](https://huggingface.co/datasets/mlpc-lab/YTTB-VQA).

<!-- ## Contents
- [Install](#installation)
- [Evaluation](#evaluation) -->

## Installation

1. Creating conda environment

```bash
conda create -n bliva python=3.9
conda activate bliva
```

2. build from source

```bash
git clone https://github.com/mlpc-ucsd/BLIVA
cd BLIVA
pip install -e .
```

## Prepare Weight

1. BLIVA Vicuna 7B

    Our Vicuna version model is released at [here](https://huggingface.co/mlpc-lab/BLIVA_Vicuna). Download our model weight and specify the path in the model config [here](bliva/configs/models/bliva_vicuna7b.yaml#L8) at line 8. 

    The LLM we used is the v0.1 version from Vicuna-7B. To prepare Vicuna's weight, please refer to our instruction [here](PrepareVicuna.md). Then, set the path to the vicuna weight in the model config file [here](bliva/configs/models/bliva_vicuna7b.yaml#L21) at Line 21.

2. BLIVA FlanT5 XXL (Available for Commercial Use)

    The FlanT5 version model is released at [here](https://huggingface.co/mlpc-lab/BLIVA_FlanT5). Download our model weight and specify the path in the model config [here](bliva/configs/models/bliva_flant5xxl.yaml#L8) at line 8. 

    The LLM weight for Flant5 will automatically begin to download from huggingface when running our inference code. 

## Inference 

To answer one question from the image, run the following evaluation code. For example,

```Shell
python evaluate.py --answer_qs \
        --model_name bliva_vicuna \
        --img_path images/example.jpg \
        --question "what is this image about?"
```

We also support answer multiple choice question, which is the same as we used for evaluation tasks in paper. To provide a list of chioce, it should be a string split by comma. For example,

```Shell
python evaluate.py --answer_mc \
        --model_name bliva_vicuna \
        --img_path images/mi6.png \
        --question "Which genre does this image belong to?" \
        --candidates "play, tv show, movie"
```
## Demo

Our Demo is publicly available at [here](https://huggingface.co/spaces/mlpc-lab/BLIVA). To run our demo locally on your machine. Run:

```Shell
python demo.py
```

## Citation

If you find BLIVA useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{hu2023bliva,
      title={BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions}, 
      author={Wenbo Hu and Yifan Xu and Yi Li and Weiyue Li and Zeyuan Chen and Zhuowen Tu},
      publisher={arXiv:2308.09936},
      year={2023},
}
```

## Acknowledgement
- [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) The model architecture of BLIVA follows BLIP-2. Don't forget to check this great open-source work if you don't know it before. 
- [Lavis](https://github.com/salesforce/LAVIS) The codebase we built upon.
- [Vicuna](https://github.com/lm-sys/FastChat) Vicuna-13B demonstrates fantastic language ability and it's open source. 

## License
This repository's code is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with 
BSD 3-Clause License [here](LICENSE_LAVIS.md).

For our model parameters of BLIVA Vicuna Version, it's should be used under LLaMA's [model license](https://github.com/facebookresearch/llama/blob/llama_v1/LICENSE). 
For the model weight of BLIVA FlanT5, it's under [Apache 2.0 License](LICENSE_BLIVA_FLANT5_WEIGHT.md). 
For our YTTB-VQA data, it's under [CC BY NC 4.0](LICENSE_DATA.md)

[![Code License](https://img.shields.io/badge/Code%20License-BSD_3--Clause-blue.svg)](LICENSE.md)
[![BLIVA Vicuna Weight License](https://img.shields.io/badge/BLIVA%20Vicuna%20Weight%20License-Non_commercial_bespoke_license-orange.svg)](https://github.com/facebookresearch/llama/blob/llama_v1/LICENSE)
[![BLIVA FLANT5 Weight License](https://img.shields.io/badge/BLIVA%20FLANT5%20Weight%20License-Apache_2.0-green.svg)](LICENSE_BLIVA_FLANT5_WEIGHT.md)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](LICENSE_DATA.md)

