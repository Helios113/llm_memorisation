---
configs:
- config_name: pistol_data_1
  data_files: sample_data_1.json
- config_name: pistol_data_2
  data_files: sample_data_2.json
license: odc-by
task_categories:
- question-answering
language:
- en
tags:
- legal
- structural unlearning
- machine unlearning
- knowledge graph
size_categories:
- n<1K
---

# Dataset Card for Dataset Name

This dataset provides a structural unlearning task for LLMs.

# Dataset Abstract

Recently, machine unlearning, which seeks to erase specific data stored in the pre-trained or fine-tuned models, has emerged as a crucial protective measure for LLMs. However, unlearning approaches for LLMs that have been considered thus far have focused on the removal of independent data points and have not taken into account that the stored facts are logically connected to one another and form an implicit knowledge graph. To facilitate the development of **structural** unlearning methods, which are essential for the practical application of unlearning, we propose PISTOL, a pipeline for compiling multi-scenario datasets for benchmarking structural LLM unlearning. Additionally, leveraging sample datasets synthesized using PISTOL, we conducted benchmarks with four distinct unlearning methods on both Llama2-7B and Mistral-7B models. This analysis helps to illustrate the prevailing challenges in effectively and robustly removing highly inter-connected data, batched data, or data skewed towards a specific domain. It also highlights the choice of pre-trained model can impact unlearning performance. This work not only advances our understandings on the limitation of current LLMs unlearning methods and proposes future research directions, but also provides a replicable framework for ongoing exploration and validation in the field.

## Dataset Details

As explained and discussed in our paper, we are providing a dataset compilation pipeline for structural unlearning dataset. This work intends to address the limitations of previous datasets, especially the lack of a structural dataset that reflects knowledge-graph-based data topology. By developing an easy-to-use and versatile pipeline, researchers can easily design and synthesize their own datasets for investigating the role that data structure or topology plays in the process of unlearning. While we leave to future research the creation of a specific large and complex dataset, we intend to advance the understanding of LLM unlearning methods by answering the questions above. We provide both sample_data_1 and sample_data_2 used in the paper. For more details regarding the data generation pipeline, please kindly refer to our paper and the [Github repo](https://github.com/bill-shen-BS/PISTOL).


- **Curated by:** [Cambridge Machine Learning Systems Lab (CaMLSys)](https://mlsys.cst.cam.ac.uk/)
- **arXiv paper:** [Paper Link](https://arxiv.org/abs/2406.16810)
- **Language(s) (NLP):** English
- **License:** ODC