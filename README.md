**Course**: CS5720 Neural Networks and Deep Learning  
**University**: University of Central Missouri  
**Student Name**: sharath chandra seriyala
**Student ID**: 700776646
---------------------------------------------------------------------------------------------------------

## 📌 Overview

This bonus assignment includes two deep learning tasks:

1. **Question Answering with Transformers** – using Hugging Face Transformers to build a QA system with pre-trained models.
2. **Digit-Class Controlled Image Generation with Conditional GAN** – building a Conditional GAN (cGAN) to generate digits from the MNIST dataset based on class labels (0–9).

----------------------------------------------------------------------------------------------------------

## Part 1: Question Answering with Transformers

### ✅ Tasks Completed

- Used Hugging Face’s `pipeline` to set up a question answering system.
- Evaluated answers using both default and custom models (`deepset/roberta-base-squad2`).
- Created a custom context and tested with multiple questions.

### Libraries Used

```bash
transformers
torch

### Commands Used
pip install transformers torch
python qa_transformer.py
----------------------------------------------------------------------------------------------------------
Part 2: Digit-Class Controlled Image Generation with Conditional GAN
✅ Tasks Completed
- Modified a GAN to accept digit labels as inputs.
- Used label embeddings in both Generator and Discriminator.
- Trained on MNIST dataset.
- Generated images for each digit class from 0 to 9.

### Libraries Used
bash
Copy
Edit
tensorflow / keras
numpy
matplotlib
▶️ How to Run
bash
Copy
Edit
pip install tensorflow numpy matplotlib
python cgan_mnist.py
📊 Output
Visualized generated digits in rows, each row corresponding to label 0 through 9.
Generator successfully learned to control digit generation by label.
Loss graphs were plotted to observe training dynamics.

📈 Result Visualization
-----------------------------------------------------------------------------------------------------------
1. How does a Conditional GAN differ from a vanilla GAN?
A Conditional GAN (cGAN) extends the vanilla GAN by introducing conditioning variables (e.g., class labels or text descriptions) to both the generator and the discriminator. This allows the generator to produce outputs based on specific input conditions, and helps the discriminator evaluate if the output not only looks real but also matches the condition.

🔍 Real-world application:
In facial attribute editing, a cGAN can generate images of a person smiling, wearing glasses, or with specific hairstyles by conditioning on attribute labels like “smiling” or “blond hair.” This enables precise control over the generation process.

2. What does the discriminator learn in an image-to-image GAN?
In an image-to-image GAN, the discriminator learns to distinguish between real and fake image pairs — that is, whether the generated image correctly corresponds to the input image it was conditioned on.

🎯 Why pairing is important:
Pairing ensures the discriminator doesn't just evaluate realism, but also the correctness of the relationship between input and output. Without correct pairing, the model could generate realistic images that don't match the input, defeating the purpose of image-to-image tasks like semantic segmentation or style transfer.
------------------------------------------------------------------------------------------------------

