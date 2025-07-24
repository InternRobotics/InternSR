# InternSR
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-orange?style=flat&logo=gradio)](#)
[![doc](https://img.shields.io/badge/Document-FFA500?logo=readthedocs&logoColor=white)](#)
[![GitHub star chart](https://img.shields.io/github/stars/InternRobotics/InternSR?style=square)](#)
[![GitHub Issues](https://img.shields.io/github/issues/InternRobotics/InternSR)](#)
<a href="https://cdn.vansin.top/taoyuan.jpg"><img src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white" height="20" style="display:inline"></a>
[![Discord](https://img.shields.io/discord/1373946774439591996?logo=discord)](https://discord.gg/5jeaQHUj4B)

## 🏠 Introduction

InternSR is an open-source toolbox for studying spatial reasoning capabilities of LVLMs based on PyTorch.

### Highlights

- High-quality Challenging Benchmarks

InternSR supports our latest challenging benchmarks with high-quality human annotations for evaluating the spatial capabilities of LVLMs, covering different inputs and scenarios.

- Easy to Use

The evaluation part is built upon VLMEvalKit, inheriting its one-command convenience for using different models and benchmarks.

- Focus on Vision-based Embodied Spatial Intelligence

Currently, InternSR focuses on spatial reasoning from ego-centric raw visual observations. Thus, it gets rid of 3D inputs and supports commonly used 2D LVLMs, meanwhile highlighting the applications in embodied interaction. We plan to support more models and benchmarks along this line.

## 🔥 News
- [2025/07] - InternSR v0.1.0 released.

## 📋 Table of Contents
- [🏠 Introduction](#-introduction)
- [🔥 News](#-news)
- [📦 Overview of Benchmark and Model Zoo](#-benchmark-model-zoo)
- [🔧 Installation](#-installation)
- [📚 Getting Started](#-getting-started)
- [🏆 Leaderboard](#-leaderboard)
- [👥 Contribute](#-contribute)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 📦 Overview of Benchmark and Model Zoo

| Benchmark       | Focus | Method | Input Modality | Data Scale |
|-----------------|-------|--------|----------------|------------|
| [MMScan](https://tai-wang.github.io/mmscan/) | Spatial Understanding | InternVL, LLaVA, QwenVL, Proprietary Models, LLaVA-3D | ego-centric videos |  300k         |
| [OST-Bench](https://rbler1234.github.io/OSTBench.github.io/) | Online Spatio-temporal Reasoning | InternVL, LLaVA, QwenVL, Proprietary Models | ego-centric videos | 10k            |
| [MMSI-Bench](https://runsenxu.com/projects/MMSI_Bench/)    | Multi-image Spatial Reasoning| InternVL, LLaVA, QwenVL, Proprietary Models | multi-view images | 1k            |
| [EgoExo-Bench](https://github.com/ayiyayi/EgoExoBench/tree/main)  | Ego-Exo Cross-view Spatial Reasoning| InternVL, LLaVA, QwenVL, Proprietary Models | ego-exo cross-view videos| 7k               |

## 🛠️ Installation

```shell
git clone https://github.com/InternRobotics/InternSR.git
cd InternSR
pip install .
```

## 📚 Getting Started

### Data preparation

We recommend placing all data under `data/`. The expected directory structure under `data/` is as follows :

```shell
data/
├── images/ # `images/` folder stores all image modality files from the datasets
├── videos/ # `videos/` folder contains all video modality files from the datasets
├── annotations/ # `annotations/` folder holds all text annotation files from the datasets
```

- #### MMScan
    1. Download the image zip files from [Hugging Face](https://huggingface.co/datasets/rbler/MMScan-2D/tree/main) (~56G), combine and unzip them under `./data/images/mmscan`.
    2. Download the annotations from [Hugging Face](https://huggingface.co/datasets/rbler/MMScan-2D/tree/main) and place them under `./data/annotations`.
    ```shell
    data/
    ├── images/
    │   ├── mmscan/
    │   │   ├── 3rscan
    │   │   ├── 3rscan_depth
    │   │   ├── matterport3d
    │   │   ├── scannet
    ├── annotations/
    │   ├── embodiedscan_video_meta/
    │   ├── ├── image.json
    │   ├── ├── depth.json
    │   ├── ├── ...
    │   ├── mmscan_qa_val_0.1.json
    │   ├── ...
    ```
    **Note**: The file `mmscan_qa_val_{ratio}.json` contains the validation data at the specified ratio.


- #### OST-Bench
    Download the images from [Hugging Face](https://huggingface.co/datasets/rbler/OST-Bench)/[Kaggle](https://www.kaggle.com/datasets/jinglilin/ostbench/)(~5G) and download the [`.tsv` file](https://opencompass.openxlab.space/utils/VLMEval/OST.tsv) , place them as follows:
    ```shell
    data/
    ├── images/
    │   ├── OST/
    │   │   ├── <scan_id>
    │   │   ├── ...
    ├── annotations/
    │   ├── OST.tsv
    ```

- #### MMSI-Bench
    Download the [`.tsv` file](https://huggingface.co/datasets/RunsenXu/MMSI-Bench/resolve/main/MMSI_bench.tsv) (~1G, including images) , place it as follows:
    ```shell
    data/
    ├── annotations/
    │   ├── MMSI_Bench.tsv
    ```
- #### EgoExo-Bench
    1. Download the processed video data from the [Hugging Face](https://huggingface.co/datasets/onlyfaces/EgoExoBench/tree/main). 
    2. Due to license restrictions, data from the [Ego-Exo4D](https://ego-exo4d-data.org/) project is not included. Users should acquire it separately by following the official Ego-Exo4D guidelines.
    3. Download the [`.tsv` file](https://huggingface.co/datasets/Heleun/EgoExoBench_MCQ/tree/main) , place them as follows: 
    ```shell
    data/
    ├── videos/
    │   ├── EgoExo4D/tasks
    │   ├── processed_frames
    │   ├── processed_video
    ├── annotations/ EgoExoBench_MCQ.tsv
    ```

### Run Evaluation

### Spatial Reasoning Benchmarks: MMSI-Bench, OST-Bench, EgoExo-Bench
Our evaluation framework for these benchmarks is built on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). The system supports evaluation of multiple model families including: o1/o3, GPT series, Gemini series, Claude series, InternVL series, QwenVL series and LLaVA series. You need to first configure the environment variables in `.env`:
```shell
OPENAI_API_KEY= 'XXX'
GOOGLE_API_KEY = "XXX"
LMUData = "./data" # the relative/absolute path of the `data` folder.
```
Available models and their configurations can be modified in `eval_tool/config.py`. To evaluate models on MMSI-Bench/OST-Bench/EgoExo-Bench, execute the following commands:
```shell
# for VLMs that consume small amounts of GPU memory
torchrun --nproc-per-node=4 scripts/run.py --data MMSI_Bench/OST/EgoExoBench_MCQ --model model_name

# for very large VLMs
python scripts/run.py --data MMSI_Bench/OST/EgoExoBench_MCQ --model model_name
```
**Note**: 
- When evaluating QwenVL-7B on EgoExo-Bench, use model_name "Qwen2.5-VL-7B-Instruct-ForVideo" instead of "Qwen2.5-VL-7B-Instruct".
- We support the Interleaved Evaluation version of OST-Bench. For the Multi-round version, please refer to [the official repository](https://github.com/OpenRobotLab/OST-Bench).

### Spatial Understanding Benchmark: MMScan

We provide two versions of the MMScan benchmark. For the 3D version, we supply RGB videos with depth information, along with camera parameters for each frame as input. The corresponding object prompts are provided in the form of 3D bounding boxes. For the 2D version, we provide RGB videos, with the corresponding object prompts given as the projected center of the object in each image.

(1) (3D Version)
We only support LLaVA-3D for the MMScan 3D Version. To run LLaVA-3D on MMScan, download the [model checkpoints](https://huggingface.co/ChaimZhu/LLaVA-3D-7B) and execute the following command:
```shell
# Single Process
bash scripts/llava3d/llava_mmscan_qa.sh --model-path path_of_ckpt --question-file ./data/annotations/mmscan_qa_val_{ratio}.json --question-file path_to_save --num-chunks 1 --chunk_idx 1

# Multiple Processes
bash scripts/llava3d/multiprocess_llava_mmscan_qa.sh
```
(2) (2D Version) For the 2D version, execute the following command to generate the results in `.xlsx` format:
```shell
# for VLMs that consume small amounts of GPU memory
torchrun --nproc-per-node=4 scripts/run.py --data MMScan_2d --model model_name

# for very large VLMs
python scripts/run.py --data MMScan_2d --model model_name
```

After obtaining results, use MMScan evaluators:
```shell
# Traditional Metrics (3D Version)
python -m scripts.eval_mmscan_qa --answer-file path_of_result

# GPT Evaluator (2D/3D Version)
python -m scripts.eval_mmscan_gpt --answer-file path_of_result --api_key XXX --tmp_path tmp_path_to_save
```

## 🏆 Leaderboard

| Models | OST-Bench | MMSI-Bench | EgoExo-Bench | MMScan(2D) | MMScan(3D) |  
|---|---|---|---|---|---|
| GPT-4o | 51.19 | 30.3 | 38.5 | 43.97 | - |
| GPT-4.1 | 50.96 | 30.9 | - | - | - |
| Claude-3.7-sonnet | - | 30.2 | 32.8 | - | - |
| QwenVL2.5-7B | 41.07 | 25.9 | 32.8 | - | - |
| QwenVL2.5-32B | 47.33 | - | 39.7 | - | - |
| QwenVL2.5-72B | - | 30.7 | 44.7 | - | - |
| QwenVL2.5-7B | 41.07 | 25.9 | 32.8 | 39.53 | - |
| QwenVL2.5-32B | 47.33 | - | 39.7 | - | - |
| QwenVL2.5-72B | - | 30.7 | 44.7 | - | - |
| InternVL2.5-8B | 47.94 | 28.7 | - | 39.36 | - |
| InternVL2.5-38B | - | - | - | 46.02 | - |
| InternVL2.5-78B | 47.94 | 28.5 | - | - | - |
| InternVL3-8B | - | 25.7 | 31.3 | 44.97 | - |
| InternVL3-38B | - | - | - | - | - |
| LLaVA-OneVision-7B | 34.92 | - | 29.5 | 39.36 | - |
| LLaVA-OneVision-72B | 44.59 | 28.4 | - | - | - |
| LLaVA-3D| - | - | - | - | 46.35 |

**Note** : 
- Different `transformers` versions may cause output variations within ±3% for the same model.
- For more detailed results, please refer to the original repositories/papers of these works.
## 👥 Contribute

We appreciate all contributions to improve InternSR. Please refer to our [contribution guide]() for the detailed instruction. For new models and benchmarks support based on VLMEvalKit, the user can also refer to the [guideline from VLMEvalKit](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Development.md).

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@misc{internsr2025,
    title = {},
    author = {InternSR Contributors},
    howpublished={\url{https://github.com/InternRobotics/InternSR}},
    year = {2025}
}
```

## 📄 License

This work is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License </a><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.

## 👏 Acknowledgement

- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): The evaluation code for OST-Bench, MMSI-Bench, EgoExo-Bench is based on VLMEvalKit.
