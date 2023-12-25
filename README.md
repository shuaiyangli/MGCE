# Multimodal Graph Causal Embedding for Multimedia-based Recommendation


<p align="left">
    <img src='https://img.shields.io/badge/key word-Recommender Systems-green.svg' alt="Build Status">
    <img src='https://img.shields.io/badge/key word-Multimodal User Preference-green.svg' alt="Build Status">
    <img src='https://img.shields.io/badge/key word-Multimodal causal embedding-green.svg' alt="Build Status">
</p>

Multimedia-based recommendation (MMRec) models typically rely on observed user-item interactions and the multimodal content of items, such as visual images and textual descriptions, to predict user preferences. Among these, the user's preference for the displayed multimodal content of items is crucial for interacting with a particular item. We argue that users' preference behaviors (i.e., user-item interactions) for the modality content of items, beyond stemming from their real interest in the modality content, may also be influenced by their conformity to the popularity of items' modality-specific content (e.g., a user might be motivated to interact with a lipstick due to enthusiastic discussions among other users regarding textual reviews of the product). In essence, user-item interactions are jointly triggered by both real interest and conformity. However, most existing MMRec models primarily concentrate on modeling users' interest preferences when capturing multimodal user preferences, neglecting the modeling of their conformity preferences, which results in sub-optimal recommendation performance. In this work, we resort to causal theory to propose a novel MMRec model, termed Multimodal Graph Causal Embedding (MGCE), revealing insights into the crucial causal relations of users' modality-specific interest and conformity in interaction behaviors within MMRec scenarios. Therein, inspired by the colliding effect in causal inference and integrating the characteristics of real interest and conformity, we devise multimodal causal embedding learning networks to facilitate the learning of high-quality causal embeddings (multimodal interest and multimodal conformity embeddings) from both the structure-level and feature-level. Therefore, MGCE is capable of precisely revealing the reasons behind user interactions with an item that has multimodal content, providing more accurate recommendations. Extensive experimental results on three public datasets demonstrate the state-of-the-art performance of MGCE.

### Before running the codes, please download the [datasets](https://www.aliyundrive.com/s/V1RPArCZQYt) and copy them to the Data directory.

## Prerequisites

- Tensorflow 1.10.0
- Python 3.6
- NVIDIA GPU + CUDA + CuDNN

