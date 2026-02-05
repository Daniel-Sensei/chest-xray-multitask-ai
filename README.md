# 🏥 Automatic Medical Report Generation from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow)
![GPT-2](https://img.shields.io/badge/GPT--2-Text%20Generation-green)
![License](https://img.shields.io/badge/License-Academic-yellow)

A multi-task deep learning framework for automated analysis of paired chest X-ray images. Combines **Image Reconstruction**, **Binary Classification**, and **Automated Report Generation** using a custom encoder-decoder architecture with GPT-2 integration.

---

## 📖 Overview

This project addresses three critical tasks in medical imaging through a unified deep learning framework:

1. **Image Reconstruction** - Reconstructs frontal and lateral X-ray views using an autoencoder with skip connections
2. **Binary Classification** - Classifies patients as Normal/Abnormal based on lung pathology (84.1% accuracy)
3. **Automated Report Generation** - Generates clinical impressions using a custom GPT-2 model conditioned on image features

### Key Innovation

The framework uses a **shared encoder** based on **CheXNet** (DenseNet121 pre-trained on 112K chest X-rays) to create a unified latent space for both frontal and lateral views. This latent representation is then used across all three tasks, ensuring consistent feature learning.

---

## 🎯 Results

### Model Performance

| Task | Metric | Score |
|------|--------|-------|
| **Frontal Reconstruction** | SSIM | 0.8856 |
| **Frontal Reconstruction** | MSE | 0.000486 |
| **Lateral Reconstruction** | SSIM | 0.8593 |
| **Lateral Reconstruction** | MSE | 0.000885 |
| **Binary Classification** | Accuracy | 84.10% |
| **Text Generation** | BLEU-4 | 0.0658 |
| **Text Generation** | ROUGE-L | 0.3024 |

### Training Metrics

- **Final Training Loss**: 0.7105
- **Final Validation Loss**: 0.7391
- **Minimal overfitting gap** demonstrates good generalization

### Example Generated Report

```
Input: No

Generated: "area of XXXX scarring or atelectasis within the lingula. 
No acute pulmonary process."

With temperature=1: "opacity in the right midlung, this could reflect 
a small focus of atelectasis or infiltrate. Bibasilar airspace opacities, 
XXXX atelectasis."
```

---

## 🏗️ Architecture

<img width="1541" height="624" alt="Unknown-2" src="https://github.com/user-attachments/assets/797786ab-6609-4a82-b000-34f78a773be1" />

### Key Components

#### 1. Encoder (Shared)
- **Base Model**: DenseNet121 pre-trained on CheXNet
- **Fine-tuning**: Last 10 layers unfrozen for domain adaptation
- **Skip Connections**: Extracted from:
  - `conv1_relu`
  - `conv2_block6_concat`
  - `conv3_block12_concat`
  - `conv4_block24_concat`
  - `conv5_block16_concat`

#### 2. Decoders (Dual)
- Separate decoders for frontal and lateral reconstruction
- **Architecture**: Transposed Conv + Batch Norm + ReLU + Conv
- **Skip Connection Fusion**: Concatenation at each upsampling stage
- **Output**: Sigmoid activation for pixel values [0, 1]

#### 3. Classifier
- **Input**: Concatenated latent vectors (frontal + lateral)
- **Architecture**: FC(512) → BN → ReLU → Dropout(0.3) → FC(256) → BN → ReLU → Dropout(0.2) → FC(1)
- **Loss**: Binary Cross-Entropy

#### 4. Custom GPT-2 Model
- **Modification**: Latent space conditioning at every transformer layer
- **Embedding**: Latent vector projected to 768d (GPT-2 dimension)
- **Training**: Teacher forcing for stable convergence
- **Generation**: Custom `generate_text()` method with temperature sampling and top-k filtering

---

## 💻 Dataset & Preprocessing

### Source Data
- **Dataset**: [Indiana University Chest X-Ray Collection](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
- **Initial Images**: Paired frontal + lateral views
- **Final Dataset**: 1,884 patient pairs

### Preprocessing Pipeline

1. **Data Cleaning**
   - Excluded patients with incomplete pairs (≠2 images)
   - Removed cases missing problems or impressions
   
2. **Classification**
   - **Normal**: 1,230 pairs (65.29%)
   - **Abnormal**: 654 pairs (34.71%)
   - Focused on lung-related conditions only

3. **Lung Segmentation**
   - Applied pre-trained U-Net for lung mask generation
   - Cropped frontal images to lung region (reduces noise from bones/organs)
   - Resized to 224×224×3

4. **Normalization**
   - Simple pixel normalization: `values / 255` → [0, 1]
   - No augmentation (tested but reduced segmentation compatibility)

---

## 🛠️ Technical Implementation

### Technologies Used

| Component | Technology |
|-----------|------------|
| **Framework** | TensorFlow 2.x / Keras |
| **Pre-trained Models** | CheXNet (DenseNet121), GPT-2 (Keras Hub) |
| **Segmentation** | U-Net (lung mask generation) |
| **Text Processing** | Custom tokenizer with `<start>`, `<end>`, `<pad>` tokens |
| **Evaluation** | SSIM, MSE, BLEU (1-4), ROUGE-L |

### Custom GPT-2 Modifications

The standard Keras Hub GPT-2 cannot accept embeddings directly. Our implementation:

1. **Extracts** GPT-2 backbone layers
2. **Injects** latent space embeddings at every transformer layer via broadcasting
3. **Implements** custom `generate_text()` method with:
   - Temperature scaling
   - Top-k sampling
   - Look-ahead masking for autoregressive generation

### Loss Function

```python
Total Loss = α * Classification Loss 
           + β * Frontal Reconstruction Loss 
           + γ * Lateral Reconstruction Loss 
           + δ * Text Generation Loss
```

**Final Component Losses:**
- Classification: 0.4384
- Frontal Reconstruction: 0.2879
- Lateral Reconstruction: 0.2301
- Text Generation: 0.0131

---

## 📊 Model Architecture Details

### Transformer from Scratch

In addition to GPT-2, the project includes a **custom Transformer** built from scratch for text generation from prompts (without latent conditioning):

- **Positional Encoding**: Sinusoidal embeddings
- **Multi-Head Attention**: 8 heads, 512d model dimension
- **Feed-Forward Network**: 2048d hidden layer
- **Training**: Custom learning rate schedule with warmup

**Example Output:**
```
Input: "No"
Generated: "area of XXXX scarring or atelectasis within the lingula. 
No acute pulmonary process."
```

---

## 👥 Authors

- **Daniel Curcio**
- **Megan Macrì**
- **Ilaria Raffaela Vasile**

*University of Calabria, Department of Mathematics and Computer Science*

---

## 🎓 Academic Context

This project was developed for the **Deep Learning** course at the University of Calabria. It demonstrates:

- Multi-task learning architecture design
- Transfer learning with medical imaging datasets
- Custom transformer implementation
- Integration of vision and language models

---

## 📄 License

This project is for academic purposes only.

---

## 📚 References

1. [Indiana University Chest X-Ray Dataset](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
2. [U-Net Lung Segmentation Model](https://www.kaggle.com/models/iamtapendu/lung-segmentation-model)
3. [CheXNet Weights](https://www.kaggle.com/datasets/sinamhd9/chexnet-weights)
