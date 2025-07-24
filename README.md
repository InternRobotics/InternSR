<div align="center">
  <img src="assets/InternSR.png" width="600"/>
</div>

<div align="center">

[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-orange?style=flat&logo=gradio)](#)
[![doc](https://img.shields.io/badge/Document-FFA500?logo=readthedocs&logoColor=white)](#)
[![GitHub star chart](https://img.shields.io/github/stars/InternRobotics/InternSR?style=square)](#)
[![GitHub Issues](https://img.shields.io/github/issues/InternRobotics/InternSR)](#)
<a href="https://cdn.vansin.top/taoyuan.jpg"><img src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white" height="20" style="display:inline"></a>
[![Discord](https://img.shields.io/discord/1373946774439591996?logo=discord)](https://discord.gg/5jeaQHUj4B)

</div>

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
- [📚 Getting Started](#-getting-started)
- [📦 Overview of Benchmark and Model Zoo](#-benchmark-model-zoo)
- [👥 Contribute](#-contribute)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 📚 Getting Started

Please refer to the [documentation](https://internrobotics.github.io/user_guide/internsr/quick_start/index.html) for quick start with InternSR, from installation to evaluating supported models.

## 📦 Overview of Benchmark and Model Zoo

| Benchmark       | Focus | Method | Input Modality | Data Scale |
|-----------------|-------|--------|----------------|------------|
| [MMScan](https://tai-wang.github.io/mmscan/) | Spatial Understanding | InternVL, LLaVA, QwenVL, Proprietary Models, LLaVA-3D | ego-centric videos |  300k         |
| [OST-Bench](https://rbler1234.github.io/OSTBench.github.io/) | Online Spatio-temporal Reasoning | InternVL, LLaVA, QwenVL, Proprietary Models | ego-centric videos | 10k            |
| [MMSI-Bench](https://runsenxu.com/projects/MMSI_Bench/)    | Multi-image Spatial Reasoning| InternVL, LLaVA, QwenVL, Proprietary Models | multi-view images | 1k            |
| [EgoExo-Bench](https://github.com/ayiyayi/EgoExoBench/tree/main)  | Ego-Exo Cross-view Spatial Reasoning| InternVL, LLaVA, QwenVL, Proprietary Models | ego-exo cross-view videos| 7k               |

### 🏆 Leaderboard

| Models | OST-Bench | MMSI-Bench | EgoExo-Bench | MMScan |
|---|---|---|---|---|
| GPT-4o | 51.19 | 30.3 | 38.5 | 43.97 |
| GPT-4.1 | 50.96 | 30.9 | - | - |
| Claude-3.7-sonnet | - | 30.2 | 32.8 | - |
| QwenVL2.5-7B | 41.07 | 25.9 | 32.8 | - |
| QwenVL2.5-32B | 47.33 | - | 39.7 | - |
| QwenVL2.5-72B | - | 30.7 | 44.7 | - |
| QwenVL2.5-7B | 41.07 | 25.9 | 32.8 | 39.53 |
| QwenVL2.5-32B | 47.33 | - | 39.7 | - |
| QwenVL2.5-72B | - | 30.7 | 44.7 | - |
| InternVL2.5-8B | 47.94 | 28.7 | - | 39.36 |
| InternVL2.5-38B | - | - | - | 46.02 |
| InternVL2.5-78B | 47.94 | 28.5 | - | - |
| InternVL3-8B | - | 25.7 | 31.3 | 44.97 |
| InternVL3-38B | - | - | - | - |
| LLaVA-OneVision-7B | 34.92 | - | 29.5 | 39.36 |
| LLaVA-OneVision-72B | 44.59 | 28.4 | - | - |
| LLaVA-3D| - | - | - | 46.35* |

**Note** : 
- Different `transformers` versions may cause output variations within ±3% score for the same model.
- For more detailed results, please refer to the original repositories/papers of these works.
- \* refers to evaluating the models with pose and depth as input as well as 3D bounding boxes as prompts on MMScan.

## 👥 Contribute

We appreciate all contributions to improve InternSR. Please refer to our [contribution guide]() for the detailed instruction. For new models and benchmarks support based on VLMEvalKit, the user can also refer to the [guideline from VLMEvalKit](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Development.md).

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@misc{internsr2025,
    title = {{InternSR: InternRobotics'} open-source toolbox for vision-based embodied spatial intelligence.},
    author = {InternSR Contributors},
    howpublished={\url{https://github.com/InternRobotics/InternSR}},
    year = {2025}
}
```

If you use the specific pretrained models and benchmarks, please kindly cite the original papers involved in our work. Related BibTex entries of our papers are provided below.

<details><summary>Related Work BibTex</summary>

```BibTex
@misc{mmsibench,
    title = {{MMSI-Bench: A} Benchmark for Multi-Image Spatial Intelligence},
    author = {Yang, Sihan and Xu, Runsen and Xie, Yiman and Yang, Sizhe and Li, Mo and Lin, Jingli and Zhu, Chenming and Chen, Xiaochen and Duan, Haodong and Yue, Xiangyu and Lin, Dahua and Wang, Tai and Pang, Jiangmiao},
    year = {2025},
    booktitle={arXiv},
}
@misc{ostbench,
    title = {{OST-Bench: Evaluating} the Capabilities of MLLMs in Online Spatio-temporal Scene Understanding},
    author = {Wang, Liuyi and Xia, Xinyuan and Zhao, Hui and Wang, Hanqing and Wang, Tai and Chen, Yilun and Liu, Chengju and Chen, Qijun and Pang, Jiangmiao},
    year = {2025},
    booktitle={arXiv},
}
@inproceedings{mmscan,
    title={{MMScan: A} Multi-Modal 3D Scene Dataset with Hierarchical Grounded Language Annotations},
    author={Lyu, Ruiyuan and Lin, Jingli and Wang, Tai and Yang, Shuai and Mao, Xiaohan and Chen, Yilun and Xu, Runsen and Huang, Haifeng and Zhu, Chenming and Lin, Dahua and Pang, Jiangmiao},
    year={2024},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
}
@inproceedings{embodiedscan,
    title={{EmbodiedScan: A} Holistic Multi-Modal 3D Perception Suite Towards Embodied AI},
    author={Wang, Tai and Mao, Xiaohan and Zhu, Chenming and Xu, Runsen and Lyu, Ruiyuan and Li, Peisen and Chen, Xiao and Zhang, Wenwei and Chen, Kai and Xue, Tianfan and Liu, Xihui and Lu, Cewu and Lin, Dahua and Pang, Jiangmiao},
    year={2024},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

## 📄 License

This work is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License </a><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.

## 👏 Acknowledgement

- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit): The evaluation code for OST-Bench, MMSI-Bench, EgoExo-Bench is based on VLMEvalKit.
