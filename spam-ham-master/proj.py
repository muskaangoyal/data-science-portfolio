#!/usr/bin/env python
# coding: utf-8

#Project Information
# We will create a classifier that can distinguish spam (junk or commercial or bulk) emails from ham (non-spam) emails.
# Spam/Ham Classification
# EDA, Feature Engineering, Classifier

# Dataset Information
# In email classification, our goal is to classify emails as spam or not spam (referred to as "ham") using features generated from the text in the email.
# The dataset consists of email messages and their labels (0 for ham, 1 for spam).
# Your labeled training dataset contains 8348 labeled examples, and the test set contains 1000 unlabeled examples.
# Run the following cells to load in the data into DataFrames.
# The `train` DataFrame contains labeled data that we will use to train your model. It contains four columns:

# 1. `id`: An identifier for the training example
# 1. `subject`: The subject of the email
# 1. `email`: The text of the email
# 1. `spam`: 1 if the email is spam, 0 if the email is ham (not spam)

# The `test` DataFrame contains 1000 unlabeled emails.
# We will predict labels for these emails and submit your predictions to Kaggle for evaluation.

#Importing libraries
from client.api.notebook import Notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import fetch_and_cache_gdrive
from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import re
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style = "whitegrid",
        color_codes = True,
        font_scale = 1.5)

# 1. Load the dataset
fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'test.csv')
original_training_data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 2. Preprocessing
## a) Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()
original_training_data.head()

## b) We will check if our data contains any missing values and replace them with appropriate filler values.
## (i.e., NaN values in the `subject` or `email` columns should be replaced with empty strings).
## Note that while there are no NaN values in the `spam` column, we should be careful when replacing NaN labels.
# Doing so without consideration may introduce significant bias into our model when fitting.

original_training_data['subject'].fillna("",inplace = True)
original_training_data['email'].fillna("",inplace = True)
original_training_data.isnull().sum()

## c) Print the text of first ham and first spam email in the original training set to see the difference between the two emails that might relate to the identification of spam.
first_ham = original_training_data[original_training_data['spam'] == 0]['email'].iloc[0]
first_spam =  original_training_data[original_training_data['spam'] == 1]['email'].iloc[0]
print(first_ham)
print(first_spam)

## We notice that spam email contains a lot of tags like head, body, html, br, href etc as compared to the ham email.
## These tags could be used to differentiate between two emails and determine if an email is spam or ham.

## d) Training Validation Split
# The training data we downloaded is all the data we have available for both training models and testing the models that we train.  We therefore need to split the training data into separate training and testing datsets. Note that we set the seed (random_state) to 42. This will produce a pseudo-random sequence of random numbers that is the same for every student.
train, test = train_test_split(original_training_data, test_size=0.1, random_state=42)


### Basic Feature Engineering
'''
We would like to take the text of an email and predict whether the email is ham or spam.
This is a classification problem, so we can use logistic regression to train a classifier.
Recall that to train an logistic regression model we need a numeric feature matrix $X$ and a vector of corresponding binary labels $y$.
Unfortunately, our data are text, not numbers.
To address this, we can create numeric features derived from the email text and use those features for logistic regression.
Each row of $X$ is an email.
Each column of $X$ contains one feature for all the emails.
'''

# Create a 2-dimensional NumPy array containing one row for each email text and that row should contain either a 0 or 1 for each word in the list.
def words_in_texts(words, texts):
    '''
    Args:
        words (list-like): words to find
        texts (Series): strings to search in

    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    indicator_array  = []
    for text in texts:
        list = []
        for word in words:
            val = [1 if (word in text) else 0]
            list += val
        indicator_array.append(list)
    return indicator_array

# 3. BASIC EDA

# We need to identify some features that allow us to distinguish spam emails from ham emails.
# One idea is to compare the distribution of a single feature in spam emails to the distribution of the same feature in ham emails.
# If the feature is itself a binary indicator, such as whether a certain word occurs in the text,
# Then this amounts to comparing the proportion of spam emails with the word to the proportion of ham emails with the word.

# The following plot (which was created using `sns.barplot`) compares the proportion of emails in each class containing a particular set of words.
# ![training conditional proportions](./images/training_conditional_proportions.png "Class Conditional Proportions")

df = pd.DataFrame({
    'word_1': [1, 0, 1, 0],
    'word_2': [0, 1, 0, 1],
    'type': ['spam', 'ham', 'ham', 'ham']
})
display(Markdown("> Our Original DataFrame has some words column and a type column. You can think of each row as a sentence, and the value of 1 or 0 indicates the number of occurances of the word in this sentence."))
display(df);
display(Markdown("> `melt` will turn columns into variale, notice how `word_1` and `word_2` become `variable`, their values are stored in the value column"))
display(df.melt("type"))

## Create a bar chart like the one above comparing the proportion of spam and ham emails containing certain words.
## Choose a set of words that have different proportions for the two classes.
## Make sure to only consider emails from `train`.

train=train.reset_index(drop=True) # We must do this in order to preserve the ordering of emails to labels for words_in_texts
set_of_words=['head','href','br']
matrix = np.matrix(words_in_texts(set_of_words, train['email']))
new_df = pd.DataFrame(matrix).rename(columns={0:'head',1:'href',2:'br'})
new_df['type'] = train['spam']
new_df = new_df.melt('type')
new_df['type'] = new_df['type'].map({0:'ham',1:'spam'})
sns.barplot(x='variable', y='value', hue='type', data=new_df, ci=None);

## When the feature is binary, it makes sense to compare its proportions across classes (as in the previous question).
## Otherwise, if the feature can take on numeric values, we can compare the distributions of these values for different classes.
## ![training conditional densities](./images/training_conditional_densities2.png "Class Conditional Densities")

## Create a class conditional density plot to compare the distribution of the length of spam emails to the distribution of the length of ham emails in the training set.
df = pd.DataFrame({'length': train['email'].apply(len),'spam': train['spam']})
df = df.melt('spam')
df['spam'] = df['spam'].map({0:'ham',1:'spam'})
x=df[df['spam']=='ham']
y=df[df['spam']=='spam']
plt.figure()
plt.xlim(0,50000)
a=sns.distplot(x['value'], label='ham', hist=False)
b=sns.distplot(y['value'], label='spam', hist=False)
a.set(xlabel='Length of email', ylabel='Distribution')
plt.legend();
## We notice in general, the length of spam emails is more than the length of ham emails.

# 4. Basic Classification
##  Notice that the output of `words_in_texts(words, train['email'])` is a numeric matrix containing features for each email.
## This means we can use it directly to train a classifier!

## `X_train` should be a matrix of 0s and 1s created by using your `words_in_texts` function on all the emails in the training set.
## `Y_train` should be a vector of the correct labels for each email in the training set.

some_words = ['drug', 'bank', 'prescription', 'memo', 'private']
X_train = np.array(words_in_texts(some_words, train['email']))
Y_train = train['spam']
X_train[:5], Y_train[:5]

# Now that we have matrices, we can use to scikit-learn!
# Using the [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier.
# Train a logistic regression model using `X_train` and `Y_train`.
# Then, output the accuracy of the model (on the training data) in the cell below.

model = LogisticRegression()
model.fit(X_train, Y_train)
training_accuracy = model.score(X_train, Y_train)
print("Training Accuracy: ", training_accuracy)

# We have trained our first logistic regression model and it can correctly classify around 76% of the training data!
# We can definitely do better than this by selecting more and better features.

# 5. Evaluating Classifiers

''' The model we trained doesn't seem too shabby! But the classifier you made above isn't as good as this might lead us to believe.
    First, we are evaluating accuracy on the training set, which may provide a misleading accuracy measure, especially if we used the training set to identify discriminative features.
    In future parts of this analysis, it will be safer to hold out some of our data for model validation and comparison.

Presumably, our classifier will be used for filtering, i.e. preventing messages labeled `spam` from reaching someone's inbox. There are two kinds of errors we can make:
 - False positive (FP): a ham email gets flagged as spam and filtered out of the inbox.
 - False negative (FN): a spam email gets mislabeled as ham and ends up in the inbox.
These definitions depend both on the true labels and the predicted labels.
False positives and false negatives may be of differing importance, leading us to consider more ways of evaluating a classifier, in addition to overall accuracy.
'''

'''
Precision measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FP}}$ of emails flagged as spam that are actually spam.
Recall measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FN}}$ of spam emails that were correctly flagged as spam.
False-alarm rate measures the proportion $\frac{\text{FP}}{\text{FP} + \text{TN}}$ of ham emails that were incorrectly flagged as spam.
'''
# The following image might help:
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/700px-Precisionrecall.svg.png" width="500px">
#
'''
Note that a true positive (TP) is a spam email that is classified as spam, and a true negative (TN) is a ham email that is classified as ham.
'''

Y_train_hat = model.predict(X_train)
true_pos = np.sum(Y_train_hat & Y_train)
total_pos = np.sum(Y_train_hat)
false_neg = np.sum(Y_train) - true_pos
false_pos = total_pos - true_pos
true_neg = np.sum(Y_train==0) - false_pos
logistic_predictor_precision = true_pos/ total_pos
logistic_predictor_recall = true_pos/ (total_pos + false_neg)
logistic_predictor_far = false_pos/ (false_pos + true_neg)
print(logistic_predictor_precision, logistic_predictor_recall,logistic_predictor_far )

## ham and spam emails
ham_emails = train[train['spam'] == 0]
spam_emails = train[train['spam'] == 1]

'''
Finding better features based on the email text:
- Number of characters in the subject / body
- Number of words in the subject / body
- Use of punctuation (e.g., how many '!' were there?)
- Number / percentage of capital letters
- Whether the email is a reply to an earlier email or a forwarded email
- Number of html tags
'''

# Number of characters in the subject
def subject_char(df):
    return df['subject'].str.findall('\w').str.len()

# Number of words in the subject
def subject_words(df):
    return df['subject'].str.findall("\w+").str.len().fillna(0)

# Use of punctuation (e.g., how many '!' were there?)
def punc_exclamation(df):
    return df['email'].str.findall("!").str.len()

def punc(df):
    return df['email'].str.findall('[^A-Za-z0-9]').str.len() / df['email'].str.findall('\w+').str.len()

# Number / percentage of capital letters
def capital_letters_percentage(df):
    return (df['subject'].str.findall(r'[A-Z]').str.len() / df['subject'].str.len())

# Whether the email is a reply to an earlier email or a forwarded email
def reply_email(df):
    return df['subject'].apply(lambda x: 1 if "Re:" in x else 0)

def forward_email(df):
    return df['subject'].apply(lambda x: 1 if "Fw:" in x else 0)

# Number of html tags
def html_tag(df):
    return df['email'].str.findall("/>").str.len()

# Number of characters in the subject
sns.distplot(subject_char(spam_emails), label = 'spam', hist=False)
sns.distplot(subject_char(ham_emails), label = 'ham', hist=False)
plt.xlabel('Number of characters in Subject');
# *We can notice that both that both the spam and ham emails have a similar amount of number of characters in the subject/body.*

# Number of words in the subject
sns.distplot(subject_words(spam_emails), label = 'spam', hist=False)
sns.distplot(subject_words(ham_emails), label = 'ham', hist=False)
plt.xlabel('Number of words in Subject');
# *We can notice that both that both the spam and ham emails have a similar amount of number of words in the subject/body.*


# Number of ! punctuations in the email
sns.distplot(punc_exclamation(spam_emails), label = 'spam', hist=False)
sns.distplot(punc_exclamation(ham_emails), label = 'ham', hist=False)
plt.xlabel('Number of punctuations (!) in emails');
# *We can notice here that spam emails have a higher use of exclamation marks as compared to the ham emails.*

# Number of punctuations in the email
sns.distplot(punc(spam_emails), label = 'spam', hist=False)
sns.distplot(punc(ham_emails), label = 'ham', hist=False)
plt.xlabel('Number of punctuations in email per word');
# *We can notice here that spam emails have a higher use of punctuations per word as compared to the ham emails.*

# Number / percentage of capital letters
sns.distplot(capital_letters_percentage(spam_emails), label = 'spam', hist=False)
sns.distplot(capital_letters_percentage(ham_emails), label = 'ham', hist=False)
plt.xlabel('percentage of capital letters in Subject');
# *Again, we find that the percentage of capital letters in the subject for both the emails are similar.*

# 2. Improving word features :
# Top words in spam and ham emails to help us find better word features.

def word_bags(df):
    wordList = {}
    for email in df['email']:
        words = re.findall('\w+', email)
        for w in words:
            if (w in wordList):
                wordList[w] += 1
            else:
                wordList[w] = 1
    return wordList

spam_bag = (pd.Series(word_bags(spam_emails)) / spam_emails.shape[0]).sort_values(ascending=False).iloc[:20]
ham_bag = (pd.Series(word_bags(ham_emails)) / ham_emails.shape[0]).sort_values(ascending=False).iloc[:20]

fig, axs = plt.subplots(ncols=2)
fig.set_size_inches(8,10)
spam_bar = sns.barplot(x=spam_bag.values, y=spam_bag.index, ax=axs[0])
spam_bar.set_title("Top words in spam emails")
hams_bar = sns.barplot(x=ham_bag.values, y=ham_bag.index, ax=axs[1])
hams_bar.set_title("Top words in ham emails")

train_word_bag = (pd.Series(word_bags(train)) / train.shape[0]).sort_values(ascending=False)[:300]
train_word_bag

## Adding new words
from sklearn.linear_model import LogisticRegressionCV
def process_data_set(df):
    some_words = ['$', '!', 'body', 'html', '/>'] + train_word_bag.index.tolist()
    X_train = np.array(words_in_texts(some_words, df['email'])).astype(int)
    feature = pd.concat([subject_words(df), punc_exclamation(df), punc(df)], axis = 1).values
    X_train = np.concatenate((X_train, feature), axis=1)
    return X_train

X_train = process_data_set(train)
Y_train = train['spam']
model = LogisticRegressionCV(Cs=4, fit_intercept=True, cv=10, verbose =1, random_state=42)
model.fit(X_train, Y_train)
training_accuracy = model.score(X_train, Y_train)
print("Training Accuracy: ", training_accuracy)


# 5. Feature/Model Selection Process
'''
- I used the idea mentioned in the section moving forward.
I visualised these features like (Number of characters in the subject, Number of words in the subject, use of punctuation, percentage of capital letters, etc.
I also digged into the email text itself to find words which could be used to distinguish between the emails. I have shown the process in the previous part.

- While plotting, I compared the distribution of the feature in ham and spam emails.
A lot of the features had similar distributions. For example, features inlcuding number of words in subjects, number of characters in the subject and number of capital letters had similar distribution for both the ham and spam emails.
While distribution of features like punctuation (!) and general punctuations were different for the ham and spam emails which means these features were good features.
I also found better words to distinguish between the emails using word bag method and inquiring the emails.
Some of these words include '$', '!', 'body', 'html', '/>', 'http', 'com', etc.

- It is suprising to see opposite distribution of general use of punctuation in the emails and specific use of exclamation marks in the emails.
Basically, we notice that ham emails use more punctuations in the emails as compared to spam emails.
We notice the opposite effect where significantly higher exclamation marks are utilised by spam emails as compared to the ham emails.
'''

# I have used wordCloud library on spam and ham emails to visualise which words are used more.
# We can notice that spam emails use words like font, html, td, tr, etc while the ham emails use words like https, com, etc.
# We can use this visualisation to choose better word features to distinguish between the spam and ham emails.

ham_emails = train[train['spam'] == 0]
spam_emails = train[train['spam'] == 1]
spam_text = spam_emails['email'].values
ham_text = ham_emails['email'].values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(spam_text))
print("SPAM EMAILS")
fig = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud)

wordcloud1 = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(ham_text))
print("HAM EMAILS")
fig1 = plt.figure(
    figsize = (40, 30),
    facecolor = 'k',
    edgecolor = 'k')
plt.imshow(wordcloud1)

plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


## 5. Submitting to Kaggle
test_predictions = model.predict(process_data_set(test))
# The following saves a file to submit to Kaggle.
submission_df = pd.DataFrame({
    "Id": test['id'],
    "Class": test_predictions,
}, columns=['Id', 'Class'])
timestamp = datetime.isoformat(datetime.now()).split(".")[0]
submission_df.to_csv("submission_{}.csv".format(timestamp), index=False)
print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
print('You may now upload this CSV file to Kaggle for scoring.')

## We got a 99.7% accuracy.
