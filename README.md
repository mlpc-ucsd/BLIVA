# BLIVA: Enhanced Vision Language Model for Understanding Text Images with Instruction Tuning
[Wenbo Hu*](https://gordonhu608.github.io/), [Yifan Xu*](https://yfxu.com/), [Yi Li](https://jerryli1019.github.io/jerryliyi.github.io/), [Weiyue Li](https://weiyueli7.github.io/), [Zeyuan Chen](https://zeyuan-chen.com/), and [Zhuowen Tu](https://pages.ucsd.edu/~ztu/). *Equal Contribution

**UC San Diego**, **Coinbase Global, Inc.**

<a href='https://gordonhu608.github.io/bliva/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>  <a href='https://arxiv.org/abs'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a> <a href='https://huggingface.co/spaces/gordonhu/BLIVA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> <a href='https://huggingface.co/mlpc-lab/BLIVA_Vicuna'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-BLIVA_Vicuna_Model-blue'></a> <a href='https://huggingface.co/mlpc-lab/BLIVA_FlanT5'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-BLIVA_FlanT5_Model-blue'></a><a href='https://huggingface.co/mlpc-lab/BLIVA_FlanT5'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Youtube Thumbnails-blue'></a>

<p align="center">
    <a href="https://huggingface.co/spaces/gordonhu/BLIVA"><img src="images/detail.png" width="90%"></a> <br> Our model architecture in detail with example responses.
</p>

## Release (Work in Progress)
- [8/3] 🔥 We released **BLIVA: Enhanced Vision Language Model for Understanding Text Images with Instruction Tuning**.  Checkout the [paper](https://arxiv.org/abs) and [demo](https://huggingface.co/spaces/gordonhu/BLIVA).

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
cd bliva
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


## Citation

If you find BLIVA useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{hu2023bliva,
      title={BLIVA: Enhanced Vision Language Model for Understanding Text Images with Instruction Tuning}, 
      author={Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, Zhuowen Tu},
      publisher={},
      year={2023},
}
```

## Acknowledgement
- [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) The model architecture of BLIVA follows BLIP-2. Don't forget to check this great open-source work if you don't know it before. 
- [Lavis](https://github.com/salesforce/LAVIS) The codebase we built upon.
- [Vicuna](https://github.com/lm-sys/FastChat) Vicuna-13B demonstrates fantastic language ability and it's open source. 

## License
This repository is under [BSD 3-Clause License](LICENSE.md).
Many codes are based on [Lavis](https://github.com/salesforce/LAVIS) with 
BSD 3-Clause License [here](LICENSE_Lavis.md).