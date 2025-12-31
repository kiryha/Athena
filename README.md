# Athena Pipeline: Research and Developmnet
AI content creation pipeline

Re imagine ComfyUI. Video generation models API. 

# Learning: Stable Diffusion & Diffusion Models

This curated learning path will take you from foundational concepts in neural networks and transformers, through the internals of diffusion models, up to practical model inference and building systems like **ComfyUI**.

## 1. Core Fundamentals

### Neural Networks & Deep Learning
- [**Neural Networks Basics**](https://www.deeplearningbook.org/)
- [**Deep Learning**](https://www.coursera.org/learn/neural-networks-deep-learning/)
- [**Backpropagation Intuition**](https://colah.github.io/posts/2015-08-Backprop/)


## 2. Transformers & Text Encoding
### Illustrated Transformer
- https://jalammar.github.io/illustrated-transformer/

### CLIP (Contrastive Language–Image Pretraining)
- [**Open AI CLIP**](https://openai.com/research/clip)
- [**Hugging Face CLIP docs**](https://huggingface.co/docs/transformers/model_doc/clip)

### Tokenization
- [**Hugging Face Tokenizers**](https://huggingface.co/docs/tokenizers/)


## 3. Diffusion Models — Theory & Concepts
### Diffusion Fundamentals
- [**Lilian Weng — Diffusion Models Guide**](https://lilianweng.github.io/posts/2023-diffusion-models/)
- [**Diffusion Models Beat GANs**](https://huggingface.co/blog/diffusion-models)

### Latent Diffusion (Stable Diffusion Architecture)
- [Latent Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)

### VAEs (Variational Autoencoders)
- [VAE Intuition](https://jaan.io/what-is-variational-autoencoder/)

## 4. Model Components & Internals
### UNet Architectures
- [Original UNet paper on segmentation](https://arxiv.org/abs/1505.04597)
- [UNet explainer](https://mlfromscratch.com/unet/)

### Scheduler & Samplers
- [Hugging Face Scheduler API](https://huggingface.co/docs/diffusers/api/schedulers)
- [Karras Scheduling Paper](https://arxiv.org/abs/2308.01250)

### CFG (Classifier-Free Guidance)
- [CFG intuition + math](https://huggingface.co/blog/diffusion-guidance)
  

## 5. Practical Inference & Libraries
### Hugging Face Diffusers
- [Official Documentation](https://huggingface.co/docs/diffusers/)

### Diffusion Text-to-Image Example
- [Example Script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/stable_diffusion_text2img.py)


### Diffusers Pipeline API
- [Pipeline Overview](https://huggingface.co/docs/diffusers/api/pipelines)


## 6. Stable Diffusion Variants & Models
### SDXL Overview
- [Hugging Face SDXL Blog](https://huggingface.co/blog/stable-diffusion-xl)

### Model Fine-tunes & Merges
- [CivitAI Model Hub](https://civitai.com/)

### Realism & Hard Surface Models
- [Model Lists & Comparisons](https://shakersai.com/ai-tools/images/stable-diffusion/sdxl/) 
  

## 7. Extensions & Advanced Topics
### ControlNet
- ControlNet Paper: https://arxiv.org/abs/2302.05543
- Official Repo: https://github.com/lllyasviel/ControlNet

### LoRA (Low-Rank Adaptation)
- LoRA Paper: https://arxiv.org/abs/2106.09685

### Quantization (INT8 / INT4)
- Hugging Face Quantization Docs:  
  https://huggingface.co/docs/transformers/main/en/performance/quantization

