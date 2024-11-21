# ML-based IDS: Tree-based Algorithms

This repository contains the implementation of a **Machine Learning-based Intrusion Detection System (IDS)** utilizing tree-based algorithms. The project demonstrates the use of classification techniques to identify and prevent potential security threats in network systems.

## Features
- Application of machine learning to detect intrusions in network traffic.
- Evaluation of tree-based algorithms such as:
  - Decision Tree
  - Random Forest
  - Extra Trees
- Comprehensive preprocessing and feature engineering steps.
- Detailed analysis of the model's performance using metrics like accuracy, precision, recall, and F1-score.
- Insights into improving the IDS with advanced machine learning techniques.

## Dataset
The project uses the **CIC-IDS 2018** dataset, which provides a detailed representation of real-world network traffic, including normal and malicious activities. This dataset is widely used in IDS research for benchmarking and evaluating model performance.

### Dataset Highlights:
- Includes a variety of attacks such as DoS, DDoS, brute force, botnet, and more.
- Simulated with realistic traffic distribution over multiple days.
- Rich set of features including protocol types, packet sizes, flow durations, and more.

The dataset can be downloaded from the official [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html) website.

## Prerequisites
- Python 3.8 or above
- Miniconda/Anaconda for managing dependencies
- Libraries:
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `imblearn`
