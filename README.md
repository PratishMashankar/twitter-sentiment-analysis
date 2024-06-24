# Machine Learning based Sentiment Analysis towards Indian Ministry

## Abstract
This project presents the performance of Twitter Sentiment Analysis on the Agricultural Ministry of India and also determines the optimal AI model between Random Forest (RF) and k Nearest Neighbors (kNN) Algorithms. 'Twitter and Reddit Sentimental Analysis' dataset from Kaggle is used to train and test the RF and kNN AI models. Term Frequency-Inverse Document Frequency (TF-IDF) Vectorizer is used to extract features from the obtained dataset. The optimal AI model is determined using the library functions of scikit-learn-2.0 which perform the training and testing of the dataset. Twitter authenticated API called Tweepy is used to create the dataset required to determine the public sentiment towards the Agriculture Ministry by us. This dataset includes 6000 tweets related to the Agriculture Ministry. The Twitter Sentiment Analysis performed determines what percentage of public opinion towards the Agriculture Ministry is positive, negative and neutral.

## Link to the Document
[Machine Learning based Sentiment Analysis toward Indian Ministry](https://rdcu.be/dxkFZ)

## File Structure
The project repository is organized as follows:

- **code/**
  - **DataCollection/**
    - Contains scripts and code related to the collection of Twitter data using Tweepy.
  - **ModelSelection/**
    - Includes code for selecting and training the optimal AI model using RF and kNN Algorithms.
  - **SentimentAnalysis/**
    - Encompasses code for performing sentiment analysis on the collected dataset.
- **report/**
  - **Report.pdf**
    - Detailed documentation of the project, including methodology, results, and analysis.
  - **Paper.pdf**
    - Academic paper discussing the approach, findings, and significance of the project.
  - **Presentation.ppt**
    - Presentation slides summarizing key aspects of the project for communication purposes.
- **requirements.txt**
  - Contains the necessary dependencies for running the project. Install using `pip install -r requirements.txt`.

## Instructions for Replication
To replicate and run the project, follow these steps:

1. Clone the repository.
2. Navigate to the respective directories under `code/` for data collection, model selection, and sentiment analysis.
3. Run the scripts in the specified order to replicate the analysis.
4. Refer to the documentation in `report/` for a comprehensive understanding of the project.

**Note:** Ensure that all dependencies, including scikit-learn-2.0 and Tweepy, are installed before running the scripts.

## Cite this work as
`Bhargavi, K. & Mashankar, Pratish & Sreevarsh, Pamidimukkala & Bilolikar, Radhika & Ranganathan, Preethi. (2022). Machine Learning-Based Sentiment Analysis Towards Indian Ministry. 10.1007/978-981-16-9573-5_28. `
