# 📚 Sentiment-Analysis-of-Ecommerce-Product-Reviews-using-Transformer-based-Fine-tuning

This repository contains code for performing sentiment analysis on e-commerce product reviews using a Transformer-based fine-tuning approach. Base model used is **DistilBERT**. Both **full-finetuning** and **parameter-efficient finetuning （LoRA)** are performed, to compare the model performance, GPU memory usage and runtime efficiency.

## Project Structure
```
Sentiment-Analysis-of-Ecommerce-Product-Reviews-using-Transformer-based-Fine-tuning/
├── main.py # Main script to run the sentiment analysis
├── requirements.txt # Required Python packages
├── Womens Clothing E-Commerce Reviews.csv # Dataset
├── results/ # Folder containing output results
│ ├── performance per epoch.png
│ └── runtime_efficiency.png
└── grid_search.ipynb # Grid search for full-finetuning and LoRA
```

## Reproduction Steps on Google Colab
Follow these steps to reproduce the results:
### 1. Open a new notebook in Google Colab.
### 2. Change the runtime type to T4 GPU:
Click Runtime → Change runtime type → Hardware accelerator → GPU T4.
### 3. Clone the repository:
`!git clone https://github.com/wenwenerrr/Sentiment-Analysis-of-Ecommerce-Product-Reviews-using-Transformer-based-Fine-tuning.git`

This should create a folder on the left panel, containing all resources from the git.
### 4. Change directory to the repository folder:
`%cd Sentiment-Analysis-of-Ecommerce-Product-Reviews-using-Transformer-based-Fine-tuning`
### 5. Install required packages:
`!pip install -r requirements.txt`
### 6. Run the main script:
`!python Main.py`

This should show training and evaluation results of both models, taking ~5 minutes.
