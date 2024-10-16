# llm_memorisation
Federated Memorisation of Fine-tuned LLMs
This project aims to evaluate model memorisation in a federated setting, focusing on fine-tuned Language Models (LLMs).
Project Overview
We're investigating the memorisation properties of fine-tuned LLMs in a federated learning context, with a particular emphasis on medical domain datasets. Our work spans several key areas:

Dataset preparation
Model fine-tuning
Memorisation attack implementation
Differential privacy application
Analysis and paper writing

Project Components
1. Dataset
We're using the following datasets:

medalpaca/medical_meadow_medical_flashcards (deduplicated version)
PISTOL dataset

2. Model Fine-tuning
We're using the Pythia Scaling suite for our experiments, with model sizes ranging from 1B to 12B parameters.
3. Memorisation Attack Implementation
We're implementing Membership Inference Attacks (MIA) using the mimir framework, adapting it to work with generic Hugging Face checkpoints.
4. Differential Privacy
We're exploring differential privacy techniques using:

Opacus
Flower's differential privacy implementation

5. Analysis and Paper Writing
Our analysis will focus on the interplay between model size, fine-tuning approaches, and memorisation in a federated learning context. We're particularly interested in the effects of using non-deduplicated base models.
