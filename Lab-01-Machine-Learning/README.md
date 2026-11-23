# Lab 01: Sentiment Analysis Using Traditional Machine Learning

## 1. Laboratory Session

Laboratory Number: 01
Title: Sentiment Analysis Using Traditional Machine Learning
Module: PUSL3205 – Large Language Models and Generative AI
Programming Language: Python

[Lab01_Sentiment_Analysis.ipynb](notebook/Lab01_Sentiment_Analysis.ipynb)

### Dataset
The Sentiment Labelled Sentences dataset from the UCI Machine Learning Repository was used by me for this task . It contains 3,000 short reviews from Amazon, IMDB and Yelp and each labelled as positive or negative.

## 2. Objective

- The purpose of this practical was to practise fundamental Natural Language Processing (NLP) tasks using sentiment analysis.
- The practical focus was around text preprocessing, converting text to numerical conversion using Bag of words (BOW), training a   traditional machine learning model and evaluating its performance.

## 3. Tasks and Implementation

The following tasks were done as per the practical sheet.

### Task 1: Download and Load the Dataset

The Sentiment Labelled Sentences dataset was sourced from the UCI Machine Learning Repository .
Three text files were used: Amazon, IMDb, and Yelp reviews. Using `read_csv()` with tab separation, the datasets were loaded into Pandas DataFrames. Then the three datasets were concatenated together using `pd.concat()`.

### Task 2: Basic preprocessing of text

Basic text preprocessing was applied to prepare the reviews for analysis.
The following operations were employed:
- lowercase text.
- Regular expressions to clean unnecessary characters and punctuation.
- Removed additional whitespace.
- Added a new column called clean_text.
- The original reviews and preprocessed text are compared.

### Task 3: Explore the Text Data

- The dataset was analysed to find the positive and the negative reviews.
- Examples of positive reviews (label 1) are shown.
- Example of negative reviews (label 0).
- Counter for identifying frequently occuring words.
- The words that appeared often in the positive and negative reviews were compared.

### Task 4: Split the Data Set

The data set was split into training and testing sets using `train_test_split()`. i set the test size to 20% and fixed the random_state to 42 for reproducibility of the split.

### Task 5: Build Bag of Words Representation

The preprocessed text was converted into numerical features using the `CountVectorizer` class of Scikit-learn.
The vectorizer was fit on the training data, and then used to transform the test data.

### Task 6: Train a Machine Learning Model

A Logistic Regression model was selected and trained with the Bag of Words features of the training dataset.
The model was specified with `max_iter=1000`.

### Task 7: Evaluate Performance of Model

The trained model was evaluated on the test dataset.
The following measures were calculated:
- Precision 
- Accuracy
- Recll
- F1 measure

### Task 8: Evaluate the Model on a New Sentence

The model, having been trained was shown a new sentence written by me.
The sentence was cleaned with the same cleaning steps , a Bag of Words representation was created and the model was used to predict the sentiment .

## 4. Results and Evidence

### Dataset Info

| Dataset | Number of Reviews Loaded |
|---|---:|
| Amazon | 1,000 |
| IMDb | 748 |
| Yelp | 1,000 |
| **Total** | **2,748** |

### Training and Testing Split

| Item | Result |
|---|---:|
| Training samples | 2,198 |
| Testing samples | 550 |
| Bag of Words vocabulary size | 4,574 |

### Model Evaluation Results

The Logistic Regression model achieved the following results:

| Evaluation Metric | Score |

| Accuracy | 82.18% |
| Precision | 81.08% |
| Recall | 81.08% |
| F1-score | 81.08% |

# New Sentence Predictions

The sentence that was used to test the model was:
- input : "I am happy with this product and it is really good"
- predicted Sentiment: Positive

supporting evidance 
- The Jupyter Notebook contains the code of implementation and the output of execution of all eight practical tasks.
- The folder screenshots contains some supporting screenshots of the implementation and results .

## 5. observations

- The text preprocessing enabled to prepare the customer reviews for the further analysis by transforming them into a consistent format.
- The most common words were common English words, like "the", "and", "I", "a" and "it".
- The text reviews were transformed into numerical features using Bag of Words which can be used by the machine learning model.
- The model of Logistic Regression showed an accuracy of 82.18% on the testing dataset.
- The model correctly predicted the response to the new test sentence as positive.
- One problem during the practical was loading the IMDb data set, which returned 748 rows. The real number of loaded rows was used when merging the datasets.

## 6. Learning Outcomes

After completing this practical session I have experience in:

- Intro to Natural Language Processing and Sentiment Analysis.
- Loading and merging text datasets with Pandas.
- Text preprocessing using Python and regular expressions.
- Exploring positive and negative reviews, word frequencies.
- Bag of Words – converting text to numerical features.
- Training a Logistic Regression model and evaluating it with classification metrics.
- Test a trained Machine Learning model on a new sentence.


