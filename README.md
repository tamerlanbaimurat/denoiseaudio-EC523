# Denoising Speech from Background Noise
### EC523: Deep Learning, Spring 2026
### Tamerlan Baimurat, Punnisa Amornsirikul, Jiaxing Wang, Michael Lwe
### {baimurat, punnisa, jiaxingw, mlwe}@bu.edu

This project works to denoise audio using parts from a Convolutional Neural Network (CNN) and Attention, specifically DeltaNet and Multihead Attention architectures.  

This README is a WIP.

## Installation and Setup

The final model notebook (`EC523_Project_Final_Model.ipynb`) is designed to be run in **Google Colab**. 

### Hardware Requirements
Due to the model size and memory constraints, you will need an **A100 40GB GPU**.

### Steps to Run
1. Open `EC523_Project_Final_Model.ipynb` in Google Colab.
2. Change the runtime type to use an A100 GPU (`Runtime` > `Change runtime type` > Hardware accelerator: `A100 GPU`).
3. Download the necessary model checkpoints from our Google Drive:
   - [Download Checkpoint (Placeholder URL)](#)
4. Mount your Google Drive or upload the checkpoint directly to your Colab environment.
5. Follow the cells in the notebook to install dependencies, load the model, and run inferences.
