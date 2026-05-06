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
   - https://drive.google.com/drive/folders/1v_ujJ8AO8EWTXrdweNC5JGt5FNAeknl_?usp=drive_link
4. Mount your Google Drive or upload the checkpoint directly to your Colab environment.
5. Follow the cells in the notebook to install dependencies, load the model, and run inferences.

## File Descriptions

| Location | Task | Who |
|---|---|---|
| `github.com/tamerlanbaimurat/project-stftgen` | Upload & convert audio to STFT | Tamerlan |
| `EC523_Project_Draft_Model.ipynb` | Overall architecture first draft | Jiaxing, Punnisa |
| `Final Report Diagrams Folder` | Architecture block diagrams | Jiaxing |
| `EC523_Project_Preliminary_Model.ipynb` | Preliminary model training & testing | Tamerlan |
| `EC523_Spectrogram_Outputs.ipynb` | Spectrogram and audio output | Tamerlan |
| `SCC_VERSION_EC523_Score_Comparisons.ipynb` | Explore other test datasets | Michael |
| `EC523_Project_Final_Model.ipynb` | Model architecture optimizations | Jiaxing, Michael |
| `EC523_Project_Baselines.ipynb` | Baseline model testing | Punnisa |
| `SCC Ablation Files Folder`, `EC523_BatchNorm_Ablation.ipynb` | Ablations training and testing | Michael, Tamerlan |
| `EC523 Project Presentation.pptx` | Prepare presentation | Everyone |
