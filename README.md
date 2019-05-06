#Python Version used: 3.6.8
#Library dependencies:
* pandas 0.24.1
* re  2.2.1
* numpy 1.15.4
* scipy 1.1.0
* matplotlib 3.0.2
* seaborn 0.9.0
* nltk 3.4
* gensim 3.7.1.
* sklearn 0.20.1
* warnings
* sys
* multiprocessing
* random

#Created by: Alexander Bock

#Goal
Use word embeddings in combination with your favourite classification algorithm to predict sentiment of movie reviews. The emphasis is on combining the two and overall code quality rather than fine tuning of the algorithm for superior accuracy.


#Solution:
We need to label dataset, after that we need to check if the class are unbalanced. Following we clean the reviews and process them. We train a word embedding model
and split the dataset into a a train/test set. On which we train a random forest classifier.  

#Observations data:
(1) punctuation seem to be removed but to be sure we will remove again
(2) no html formatting found e.g <br> but to be sure we shall remove nonetheless
(3) positive and negative reviews seem to be balanced in row length but are they balanced in text length?
(4) Since we have a 50/50 split. Our benchmark for any classification  algorithm needs to beat a accuracy of 50%. Under the assumption that this ratio is an representative sample of the true distribution of positive to negative reviews within the true population
(5) Reviews seems to be drawn from the same distribution in terms of length of review
(6) We achieved a accuracy of ~70% using the trained word-embeddings. Using a pre-trained word-embedding model or using more training data is more likely to increase the performance of the.


###Coding Steps:
(1) Load data
(2) Label data
(3) Combining both datasets into one and EDA
(4) Processing of text (removal of stops words, line breakers, removal of numbers. etc.)
(5) Vectorization of text using doc2vec approach
(6) Classification model training & evaluation phase, and save to disk for future use
(7) Run with unlabelled data
(8) ?
(9) Profit

#Output files:
* Confusion_matrix as png
* Pickled Model RandomForestClassifier
* sentiment_predictions as csv
