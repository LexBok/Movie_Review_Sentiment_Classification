
# coding: utf-8
### Library loadin ###
#general lib
import pandas as pd #data handeling library
import re #regular expression library
import numpy as np #vector lib
from scipy import stats #stats lib
import warnings #turning off future warning
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import multiprocessing #for faster word embedding training
import random #getting random items


#data viz lib
import matplotlib.pyplot as plt #data viz lib
import seaborn as sns #data viz lib

#language lib for stemming,token and cleaning
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

#word/doc vector creation
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#ml lib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.externals import joblib #saving model


print(" _____            _                  _____            _   _                      _ ")
print("|  __ \          (_)                / ____|          | | (_)                    | |")
print("| |__) |_____   ___  _____      __ | (___   ___ _ __ | |_ _ _ __ ___   ___ _ __ | |_ ")
print("|  _  // _ \ \ / / |/ _ \ \ /\ / /  \___ \ / _ \ '_ \| __| | '_ ` _ \ / _ \ '_ \| __|")
print("| | \ \  __/\ V /| |  __/\ V  V /   ____) |  __/ | | | |_| | | | | | |  __/ | | | |_ ")
print("|_|  \_\___| \_/ |_|\___| \_/\_/   |_____/ \___|_| |_|\__|_|_| |_| |_|\___|_| |_|\__|")
print("")


#### Step One Load Data ####
#importing positive reviews from txt file
print("------Step One Load data-----")
try:
    positive_reviews = []
    for line in open('positive_reviews.txt', 'r'):
        positive_reviews.append(line.strip())

    #importing negative reviews from txt file
    negative_reviews = []
    for line in open('negative_reviews.txt', 'r'):
        negative_reviews.append(line.strip())

    #importing unsupervised reviews from txt file
    unsupervised_reviews = []
    for line in open('unsupervised_reviews.txt', 'r'):
        unsupervised_reviews.append(line.strip())
    print("Done Next Step")
    print("")
except:
    sys.exit("Error occured while loading txt file")



####### Step Two Label Data ######
print("------Step Two Label data-----")
try:
    #creation of pandas dataframe for fast pre-proccesing
    df_positive = pd.DataFrame(positive_reviews, columns=['review'])
    df_negative = pd.DataFrame(negative_reviews, columns=['review'])
    df_unsupervised = pd.DataFrame(negative_reviews, columns=['review'])

    df_positive['sentiment'] = "positive"
    df_negative['sentiment'] = "negative"
    df_unsupervised['sentiment'] = "unknown"
    print("Done Next Step")
    print("")
except:
    sys.exit("Error occured while creating pandas dateframe")




####### Step Three Combining both data sets into one and EDA #######
print("------Step Three Combining both data sets into one and EDA -----")
df = df_positive.append(df_negative)
#reseting index
df = df.reset_index()

print("Classes unbalanced?")
print(df.sentiment.value_counts())
print("")
#Not unbalanced in terms on size of reviews. 50/50 split.
#Thus our benchmark for any classification algorithm needs to beat a accuracy of 50%
#Under the assumption that this ratio is an representative sample
#of the true distribution of positive to negative reviews within the true population


#Total amount of words within dataset
#print(f"Sum of words in train/test dataset: {df['review'].apply(lambda x: len(x.split(' '))).sum()}")



#Null values?
print ("Null values present within sentiment dataframe?")
print(df.isnull().sum())
print("")

print ("Null values present within unsupervised dataframe?")
print(df_unsupervised.isnull().sum())
print("")

#Since classes are balanced is the length of the reviews balanced?
def cell_length(df,new_col_name,target_col):
    try:
        length = []
        for index, row in df.iterrows():
            length.append(len(df[target_col][index]))
        df[new_col_name] = length
    except:
        sys.exit("Error occured while calulating the lenght of cell")

cell_length(df,"length_review","review")
cell_length(df_unsupervised,"length_review","review")


def length_analysis():
    try:
        print("")
        print("lengths of reviews balanced within train/test set?")
        print(df.groupby('sentiment')['length_review'].describe())
        print("")
        #Sample of positive and negative reviews seem to be quite homogeneous

        #testing this hypothesis using two-sided t-test
        negative_mean = df[df.sentiment == "negative"]['length_review']
        positive_mean = df[df.sentiment == "positive"]['length_review']

        t_statistic, p_value = stats.ttest_ind(negative_mean, positive_mean)

        print("Running two sided t-test on mean of review lenght within train/test set")
        print(f"t_statistic: {t_statistic}, p_value: {p_value}")
        print("")
        #We can assume that both have been drawn from the same population.

        print("Are the reviews text lenghts in the train/test set similar compared to the unsupervised set?")
        t_statistic, p_value = stats.ttest_ind(df.length_review, df_unsupervised.length_review)
        print(f"t_statistic: {t_statistic}, p_value: {p_value}")
        #It seems like they are also drawn from the same population
        print("Done Next Step")
        print("")

    except:
        sys.exit("Error occured during lenght lenght_analysis")

length_analysis()




####### Step Four processing of text #######
print("------Step Four Processing of text-----")
def preprocess_text(text):
    try:
        #Remove punctuation#
        replace_punctuation = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
        punctuation = ""
        text = [replace_punctuation.sub(punctuation, line.lower()) for line in text]

        #Remove formating_tags#
        replace_formating_tags = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
        formating_tags = " "
        text = [replace_formating_tags.sub(formating_tags, line) for line in text]

        return text
    except:
        sys.exit("Error occurred while preprocessing text")

df.review = preprocess_text(df.review)


def process_text(raw_text):
    try:
        """Clean raw text and tokenize"""

        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')

        # remove non ASCII char
        ascii_text = re.sub(r'[^\x00-\x7F]+',' ', raw_text.lower())

        # tokenize text
        tokens = tokenizer.tokenize(ascii_text)

        # remove stop words
        tokens_no_sw = filter(lambda t: t not in stop_words, tokens)

        # replace numeric values with placeholder
        token_alpha = map(lambda t: re.sub(r'[1-9][0-9]*', '[NNNN]', t), tokens_no_sw)

        # extract stems
        token_stem = map(lambda t: stemmer.stem(t), token_alpha)

        # remove 1 char tokens
        token_non_empty = filter(lambda t: len(t) > 1, token_stem)
        token_non_empty = list(token_non_empty)


        return token_non_empty


    except:
        sys.exit("Error occurred while processing text")



df['tokenized'] = df['review'].apply(process_text)
print("Done Next Step")
print("")


####### Step Five Vectorization of text using doc2vec approach #######
print("------Step Five Vectorization of text using doc2vec approach-----")
#While the word vectors represent the concept of a word,
#the document vector intends to represent the concept of a document.
def doc_to_vec(col):
    try:
        #accessing all cores
        cores = multiprocessing.cpu_count()

        #tagging of documents
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(col)]

        #distributed bag of words (PV-DBOW) model
        model_dbow = Doc2Vec(documents,dm=0,vector_size=100, window=2, min_count=2, workers=cores, epochs=20)

        #assigning  vecotrs
        global X
        X = model_dbow.docvecs.vectors_docs
        return X


    except:
        sys.exit("Error occured while creating document vectors")


doc_to_vec(df['tokenized'])
print("Done Next Step")
print("")

####### Step Six Classification Model Training & Evaluation Phase #######
print("------Step Six Classification Model Training & Evaluation Phase-----")
def random_forest_review_classifier(X):
    """Split into training & test sets"""
    try:
        y = df['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    except:
        sys.exit("Error occured while splitting train and test set")


    """traning of model"""
    try:

        rf = RandomForestClassifier(n_estimators = 200, max_depth=10,random_state=42,min_samples_split=2,)
        print(f"Length X_train: {len(X_train)}")
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        acc_score = str(accuracy_score(y_test, y_pred)*100)
        print(f"Accuracy_score: {acc_score}%")
        print(f"Confusion_matrix: {confusion_matrix(y_test, y_pred)}")

    except:
        sys.exit("Error occured while training random forest classifier")


    """plotting confusion matrix"""
    try:
        ax = sns.heatmap(confusion_matrix(y_test, y_pred),annot=True, fmt="d",robust=True,cmap="YlGnBu")
        #labeling of confusion matrix
        class_names = ["positive","negative"]
        plt.title("Confusion Matrix Movie Reviews")
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names, rotation=45)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        sns.set(rc={'figure.figsize':(10,6)})
        fig = ax.get_figure()
        fig.savefig('confusion_matrix.png')
        print("Confusion Matrix Saved Check Folder")
    except:
        sys.exit("Error occured while plotting confusion matrix")

    """Saving model and vectors"""
    try:
        # save model to a file
        joblib.dump(rf, 'review_classifier_model.pkl')
        # load it later
        #rf = joblib.load('review_classifier_model.pkl')
        print("Done Next Step")
        print("")
    except:
        sys.exit("Error occured while saving model")



random_forest_review_classifier(X)


####### Step Seven Run with unlabelled data#######
print("------Step Seven Run with unlabelled data-----")
#load model
rf = joblib.load('review_classifier_model.pkl')


#Process data
df_unsupervised.review = preprocess_text(df_unsupervised.review)
df_unsupervised['tokenized'] = df_unsupervised['review'].apply(process_text)


#doc 2 vec
doc_to_vec(df_unsupervised['tokenized'])

#predict sentiment on dov2vec vecotrs
y_pred = rf.predict(X)

#append to dataframe
df_unsupervised['sentiment'] = y_pred
print("------------------------------------")
print("Predicted Sentiment within data set")
print(df_unsupervised.sentiment.value_counts())


def sample_test():
    random_idx = (random.randint(0,df_unsupervised.shape[0]))

    print("")
    print("Sample test")
    print("-------------")
    print(f"Review: {df_unsupervised.review[random_idx]}")
    print("-------------")
    print(f"Predicted Sentiment: {df_unsupervised.sentiment[random_idx]}")
    print("")

for i in range(0,10):
    sample_test()

#Saving predications
df_unsupervised.to_csv('sentiment_predictions_unsupervised.csv')
print("Prediction Saved Check Folder")

print("")
print("")
print(" _____  ")
print("|  __ \ ")
print("| |  | | ___  _ __   ___  ")
print("| |  | |/ _ \| '_ \ / _ \ ")
print("| |__| | (_) | | | |  __/ ")
print("|_____/ \___/|_| |_|\___| ")
