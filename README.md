# Fake News Detection

This repository contains a machine learning project focused on detecting fake news using various natural language processing (NLP) techniques and classification models. The project explores different approaches to identify and classify news articles as real or fake.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Dependencies](#dependencies)
- [Data](#data)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Introduction

Fake news has become a significant issue in today's digital age. This project aims to tackle this problem using machine learning models to classify news articles based on their content. By analyzing the text data in form of word embeddings, we can train models that effectively distinguish between real and fake news.

## Features

- NLP preprocessing on the news descriptions.
- Exploratory Data Analysis (EDA) on the dataset.
- Implementation of Word2Vec embeddings.
- Multiple machine learning models including Logistic Regression, Naive Bayes, Random Forest, and Gradient Boosting.
- Model evaluation and comparison.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/dxlee0807/fake-news-detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd fake-news-detection
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the project is set up, you can explore the notebooks and scripts provided in the repository. For example, to run the data processing and model training:

1. Open the Jupyter notebooks provided.
2. Follow the instructions in the notebook to preprocess the data and train the models.
3. Evaluate the performance of different models.

## Models Implemented

- **Logistic Regression** by [desmondsiew](https://github.com/desmondsiew)
- **Naive Bayes** by [venice0507](https://github.com/venice0507)
- **Random Forest** by [zhenmin01](https://github.com/zhenmin01)
- **Gradient Boosting** by [dxlee0807](https://github.com/dxlee0807)

Each model is trained on Word2Vec embeddings derived from the text data, and their performances are compared to find the most effective one.

## Dependencies

The project requires the following Python packages:
- `numpy`
- `pandas`
- `scikit-learn`
- `nltk`
- `gensim`
- `matplotlib`
- `seaborn`
- `jupyter`

These can be installed via the `requirements.txt` file in the repository.

## Data

The dataset used for this project includes labelled news articles, which are preprocessed and used to train the models. The details of data preprocessing are available in the notebooks.

Dataset Link: https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection

## Results

The project compares the performance of different models based on accuracy, precision, recall, F1-score, and confusion matrix. The results are summarized and visualized to highlight the best-performing model.

## Contributors

- [dxlee0807](https://github.com/dxlee0807)
- [desmondsiew](https://github.com/desmondsiew)
- [venice0507](https://github.com/venice0507)
- [zhenmin01](https://github.com/zhenmin01)

## License

This project is licensed under the MIT License.
