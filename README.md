# Intern-EP
## 🏠 Introduction

The Intern-EP repository focuses on embodied perception, covering both benchmarks and models for embodied scene understanding and reasoning. Currently, it includes:
### Models
- [LLaVA-3D](https://zcmax.github.io/projects/LLaVA-3D/)
### Benchmarks
- [MMScan](https://tai-wang.github.io/mmscan/), [OST-Bench](https://rbler1234.github.io/OSTBench.github.io/), [MMSI-Bench](https://runsenxu.com/projects/MMSI_Bench/), and EgoExo-Bench.







## 🔥 News
- [2025/07] - The first version of Intern-EP includes the model LLaVA-3D, and benchmarks MMScan, OST-Bench, MMSI-Bench, and EgoExo-Bench.

## 📋 Table of Contents
- [🏠 Introduction](#-introduction)
- [🔥 News](#-news)
- [📚 Getting Started](#-getting-started)
- [📦 Overview of Benchmark and Model Zoo](#-benchmark-model-zoo)
- [🔍 Evaluation Tutorial](#-evaluation-tutorial)
- [👏 Acknowledgements](#-acknowledgements)
- [📝 TODO List](#-todo-list)

## 📚 Getting Started
Clone this repo.
```shell
git clone https://github.com/rbler1234/Intern-EP.git
cd Intern-EP
```

<details>
<summary><b><font size="3">Installation</font></b></summary>

(a) To enable evaluation for the benchmarks, please install the following dependencies:

```shell
# For OST-Bench/MMSI-Bench/EgoExo-Bench evaluation:
pip install -r requirement/base.txt
# For MMScan evaluation:
pip install -r requirement/mmscan.txt
```
(b) To perform the inference of LLava3D, please install the required environment as follows:
```shell
cd vlm/LLaVA-3D
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install -e .
```
</details>

<details>
<summary><b><font size="3">Data preparation</font></b></summary>
We recommend placing all data under `./data`. The expected directory structure under `./data` is as follows :

```shell
./data
├── images # `images/` folder stores all image modality files from the datasets
├── videos # `videos/` folder contains all video modality files from the datasets
├── annotations # `annotations/` folder holds all text annotation files from the datasets
```

- #### MMScan
    1. Download the image zip files from [huggingface](https://huggingface.co/datasets/rbler/MMScan-2D/tree/main)(~56G), combine and unzip them under `./data/images/mmscan`.
    2. Download the annotations from [huggingface](https://huggingface.co/datasets/rbler/MMScan-2D/tree/main) and place them under `./data/annotations`.
    ```shell
    ./data
    ├── images/
    │   ├── mmscan/
    │   │   ├── 3rscan
    │   │   ├── 3rscan_depth
    │   │   ├── matterport3d
    │   │   ├── scannet
    ├── annotations/
    │   ├── embodiedscan_infos_full.json
    │   ├── mmscan_qa_val_0.1.json
    │   ├── ...
    ```
    *Note*: The file `mmscan_qa_val_{ratio}.json` contains the validation data at the specified ratio.


- #### OST-Bench
    Download the images from [huggingface](https://huggingface.co/datasets/rbler/OST-Bench)/[kaggle](https://www.kaggle.com/datasets/jinglilin/ostbench/)(~5G) and download the [`.tsv` file](https://opencompass.openxlab.space/utils/VLMEval/OST.tsv) , place them as follows:
    ```shell
    ./data
    ├── images/
    │   ├── OST/
    │   │   ├── <scan_id>
    │   │   ├── ...
    ├── annotations/ OST.tsv
    ```

- #### MMSI-Bench
    Download the [`.tsv` file](https://huggingface.co/datasets/RunsenXu/MMSI-Bench/resolve/main/MMSI_bench.tsv) (~1G, includes images) , place it as follows:
    ```shell
    ./data
    ├── annotations/ MMSI_Bench.tsv
    ```
- #### EgoExo-Bench
    1. Download the videos from the following sources: [Ego-Exo4D](https://ego-exo4d-data.org/), [LEMMA](https://sites.google.com/view/lemma-activity), [EgoExoLearn](https://huggingface.co/datasets/hyf015/EgoExoLearn), [TF2023](https://github.com/ziweizhao1993/PEN), [EgoMe](https://huggingface.co/datasets/HeqianQiu/EgoMe), [CVMHAT](https://github.com/RuizeHan/CVMHT).
    2. Download the [`.tsv` file](https://drive.google.com/file/d/1pRGd9hUgwCzMU6JSPFxpjGAtCChwIB9G/view?usp=sharing) , place them as follows: 
    ```shell
    ./data
    ├── videos/
    │   ├── CVMHAT/data
    │   ├── EgoExo4D/tasks
    │   ├── EgoExoLearn
    │   ├── LEMMA
    │   ├── TF2023/data
    ├── annotations/ EgoExoBench_MCQ.tsv
    ```
</details>

## 📦 Overview of Benchmark and Model Zoo

| Benchmark       | Domain                | Method                                      | Input Modality                                       | Data  Scale                                     |
|-----------------|-----------------------|----------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------|
| [MMScan](https://tai-wang.github.io/mmscan/)        | Scene Understanding      | [LLaVA-3D](https://zcmax.github.io/projects/LLaVA-3D/) | `point cloud`, `image`, `video`, `text`              |  300k         |
| [OST-Bench](https://rbler1234.github.io/OSTBench.github.io/)    | Scene Spatial Reasoning  | LLaVA / InternVL / QwenVL / Proprietary Models     | `video`, `text`                                      | 10k            |
| [MMSI-Bench](https://runsenxu.com/projects/MMSI_Bench/)    | Scene Spatial Reasoning       | LLaVA / InternVL / QwenVL / Proprietary Models                      | `image`, `text`                      | 1k            |
| EgoExo-Bench  | Scene Spatial Reasoning | LLaVA / InternVL / QwenVL / Proprietary Models                       | `video`,  `text`                             | 7k               |

## 🔍 Evaluation Tutorial
### For OST-Bench/MMSI-Bench/EgoExo-Bench
Our evaluation framework for these benchmarks is built on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit). The system supports evaluation of multiple model families including: o1/o3, GPT series, Gemini series, Claude series, InternVL series, QwenVL series and LLaVA series. You need to first configure the environment variables in `.env`:
```shell
OPENAI_API_KEY= 'XXX'
GOOGLE_API_KEY = "XXX"
LMUData = "./data" # the relative/absolute path of the `data` folder.
```
Available models and their configurations can be modified in `eval_tool/config.py`. To evaluate models on OST-Bench/MMSI-Bench/EgoExo-Bench, execute the following commands:
```shell
# for VLMs that consume small amounts of GPU memory
torchrun --nproc-per-node=1 scripts/run.py --data OST/MMSI_Bench/EgoExoBench_MCQ --model model_name

# for very large VLMs
python scripts/run.py --data OST/MMSI_Bench/EgoExoBench_MCQ --model model_name
```
*Note*: (1) When evaluating QwenVL-7B on EgoExo-Bench, use model_name "Qwen2.5-VL-7B-Instruct-ForVideo" instead of "Qwen2.5-VL-7B-Instruct".
(2) We support the Interleaved Version Evaluation of OST-Bench, for the Multi-round Version, refer to [code](https://github.com/OpenRobotLab/OST-Bench).

### For MMScan

We only support LLaVA-3D for the MMscan Question Answering Benchmark currently. To run LLaVA-3D on MMScan, download the [model checkpoints](https://huggingface.co/ChaimZhu/LLaVA-3D-7B) and execute the following command:
```shell
# Single Process
bash scripts/llava3d/llava_mmscan_qa.sh --model-path path_of_ckpt --question-file ./data/annotations/mmscan_qa_val_{ratio}.json --question-file path_to_save --num-chunks 1 --chunk_idx 1

# Multiple Processes
bash scripts/llava3d/multiprocess_llava_mmscan_qa.sh
```
After obtaining results, use MMScan evaluators:
```shell
# Traditional Metrics
python -m scripts.eval_mmscan_qa --answer-file path_of_result

# GPT Evaluator
python -m scripts.eval_mmscan_gpt --answer-file path_of_result --api_key XXX --tmp_path tmp_path_to_save
```

## 👏 Acknowledgement

- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): The evaluation code for OST-Bench / MMSI-Bench / EgoExo-Bench is based on VLMEvalKit.

## 📝 TODO List
- \[ \] Support more models for MMScan.
- \[ \] Support MMScan Visual Grounding Task.


