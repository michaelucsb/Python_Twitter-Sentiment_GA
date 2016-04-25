# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:48:09 2016

@author: Michael Lin_2
"""
#################################################################################################
#################################################################################################
#####
#####      
#####      DATA LOADING
#####
#####
#################################################################################################
#################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.cross_validation import train_test_split

import warnings
warnings.simplefilter(action = "ignore", category = DeprecationWarning)
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)

'''## Pandas Dataframe Display Options
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 800)

## To reset
%reset
'''
pd.set_option('display.width', 160)
pd.set_option('max_colwidth', 160)

############################################################
## 1st Datasets: My own tweets collection over 7 days period
############################################################
df_aa = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/AmericanAir_full.csv')
df_da = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/DeltaAir_full.csv')
df_sw = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/Southwest_full.csv')
df_ua = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/UnitedAir_full.csv')

cols = ['tweet_id','content', 'retweeted_status', 'coordinates', 'media_type', 'video_link', 'photo_link', 'twitpic', 'source']

df1 = df_aa[cols].copy()
df2 = df_da[cols].copy()
df3 = df_sw[cols].copy()
df4 = df_ua[cols].copy()

df1['airline'] = 'AA'
df2['airline'] = 'DA'
df3['airline'] = 'SW'
df4['airline'] = 'UA'

df_text = pd.concat([df1, df2, df3, df4])
df_text['retweet'] = 'no'
df_text.loc[df_text.retweeted_status == 'THIS IS A RETWEET --> DOUBLE-CHECK JSON', 'retweet'] = 'yes'
df_text.drop('retweeted_status', axis=1, inplace=True)
df_text.reset_index(inplace=True)
df_text.shape



#############################################################
## 2st Datasets: 2,400 manually graded tweets from collection
#############################################################
df_aa600 = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/American_600.csv', encoding='latin-1')
df_da600 = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/Delta_600.csv', encoding='latin-1')
df_sw600 = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/Southwest_600.csv', encoding='latin-1')
df_ua600 = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/United_600.csv', encoding='latin-1')
df_aa600 = df_aa600.rename(columns={'Sentiment': 'sentiment'})
df_da600 = df_da600.rename(columns={'Sentiment': 'sentiment'})
df_ua600 = df_ua600.rename(columns={'Sentiment': 'sentiment'})
df_aa600['airline'] = 'AA'
df_da600['airline'] = 'DA'
df_sw600['airline'] = 'SW'
df_ua600['airline'] = 'UA'

df_2400 = pd.concat([df_aa600, df_da600, df_sw600, df_ua600])
df_2400.sentiment = df_2400.sentiment - 2

##########################################################
## 3rd Datasets: 4,000 machine graded tweets from Newsroom
##########################################################
df_aa_nr = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/american_newsroom.csv', encoding='latin-1')
df_da_nr = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/delta_newsroom.csv', encoding='latin-1')
df_sw_nr = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/southwest_newsroom.csv', encoding='latin-1')
df_ua_nr = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/united_newsroom.csv', encoding='latin-1')
df_aa_nr['airline'] = 'AA'
df_da_nr['airline'] = 'DA'
df_sw_nr['airline'] = 'SW'
df_ua_nr['airline'] = 'UA'
df_nr_full = pd.concat([df_aa_nr, df_da_nr, df_sw_nr, df_ua_nr])
df_nr = df_nr_full[['body', 'sentiment', 'airline']].copy()
df_nr.rename(columns={'body':'content'}, inplace=True)

senti = []
for sentiment in df_nr.sentiment:
    if sentiment == 'Neutral':
        senti.append(0)
    elif sentiment == 'Positive':
        senti.append(1)
    else:
        senti.append(-1)
df_nr.sentiment = senti

######################################################
## 4th Datasets: Airline Tweets from Kaggled (Cleaned)
######################################################
df_kaggle = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/Tweets.csv')

df_kaggle.rename(columns={'text':'content'}, inplace=True)

senti = []
for sentiment in df_kaggle.airline_sentiment:
    if sentiment == 'neutral':
        senti.append(0)
    elif sentiment == 'positive':
        senti.append(1)
    else:
        senti.append(-1)
df_kaggle['sentiment'] = senti



#################################################################################################
#################################################################################################
#####
##### A class to preprocess all the tweets, both test and training
##### We will use regular expressions and NLTK for preprocessing  
#####
#####
#################################################################################################
#################################################################################################
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import string
import re

## Define stop words in English
stop = stopwords.words('english')

## Pre-processing using regular expression
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub(r'@([^\s]+)', r'\1', tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Replace '&amp;' with 'and'
    tweet = re.sub(r'&amp;', r' and', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
# END processTweet()

## Add a new column for pre-processing tweet content 
def adding_ppcontent_col(df):
    pptext=[]
    
    for line in df.content:
        text = processTweet(line)
        pptext.append(text)

    df['ppcontent'] = pptext
# END adding_ppcontent_col()

adding_ppcontent_col(df_text)
adding_ppcontent_col(df_2400)
adding_ppcontent_col(df_nr)
adding_ppcontent_col(df_kaggle)


regex = re.compile('[%s]' % re.escape(string.punctuation))
def remove_pun(s):  # From Vinko's solution, with fix.
    return regex.sub('', s)
   

## Add a new column that filters stop words for pre-processing tweets
def adding_fcontent_col(df):
    ftext = []
    
    for line in df.ppcontent:
        line = remove_pun(line)
        text = [word for word in line.split() if word not in stop]
        ftext.append(text)
    
    df['fcontent'] = ftext
# END adding_fcontent_col()

adding_fcontent_col(df_text)
adding_fcontent_col(df_2400)
adding_fcontent_col(df_nr)
adding_fcontent_col(df_kaggle)






#################################################################################################
#################################################################################################
#####
##### 
##### RULES based Sentiment Analysis
#####
#####
#################################################################################################
#################################################################################################

###########################################################################################
#### Sentiment Analysis based on the "Common Sense" Rules
#### ----------------------------------------------------
#### (pos) x (pos) = (pos)   i.e. This movie is very good
#### (pos) x (neg) = (neg)   i.e. This movie is not great
#### (neg) x (neg) = (pos)   i.e. This movie was not bad
###########################################################################################
'''may need to download new positive and negative lists'''

import nltk
import csv

negative = []
with open("words_negative.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        negative.append(row)

positive = []
with open("words_positive.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        positive.append(row)

## Look at every single sentence (separated by period ".") to classify their sentiment.
def rule_sentiment(text):
    temp = [] #
    text_sent = nltk.sent_tokenize(text)
    for sentence in text_sent:
        n_count = 0
        p_count = 0
        sent_words = nltk.word_tokenize(sentence)
        for word in sent_words:
            for item in positive:
                if(word == item[0]):
                    p_count +=1
            for item in negative:
                if(word == item[0]):
                    n_count +=1

        if(p_count > 0 and n_count == 0): #any number of only positives (+) [case 1]
            #print "+ : " + sentence
            temp.append(1)
        elif(n_count%2 > 0): #odd number of negatives (-) [case2]
            #print "- : " + sentence
            temp.append(-1)
        elif(n_count%2 ==0 and n_count > 0): #even number of negatives (+) [case3]
            #print "+ : " + sentence
            temp.append(1)
        else:
            #print "? : " + sentence
            temp.append(0)
    return temp
# END rule_sentiment()

## This function will add a new column "rule_sentiment" to the df
def adding_rule_sentiment(df):
    # Must already have 'ppcontent' column in df
    senti = []
    for line in df.ppcontent:
        senti.append(rule_sentiment(line))
    df['rule_senti'] = senti

    ## Look at all the sentiment in each tweet and classify the tweet
    senti = []
    for line in df.rule_senti:
        if -1 in line:
            senti.append(-1)
        elif sum(line) == 0:
            senti.append(0)
        else:
            senti.append(1)
    df['rule_senti'] = senti
# END adding_rule_sentiment()

## This function prints out a confusion matrix
def print_confusion(truth, prediction):
    confusion = pd.crosstab(truth, prediction, rownames=['Predicted'], colnames=['     True'], margins=True)
    accuracy = (truth == prediction).sum()/len(truth)*100
    print(confusion)
    print(accuracy)
# END print_confusion()
    
################################
## Now let's call these function
## and look at the confusion!
################################
adding_rule_sentiment(df_2400)
print_confusion(df_2400.rule_senti, df_2400.sentiment)
#      True    -1    0    1   All
# Predicted                      
# -1          605   76   46   727
# 0           510  356  138  1004
# 1           318  124  227   669
# All        1433  556  411  2400
# Rule Accuracy = 49.50%

adding_rule_sentiment(df_text)

adding_rule_sentiment(df_nr)
print_confusion(df_nr.rule_senti, df_nr.sentiment)
#      True   -1     0    1   All
# Predicted                      
# -1          97   444  121   662
# 0          340  1558  385  2283
# 1           56   556  443  1055
# All        493  2558  949  4000
# Rule Accuracy = 52.45%

adding_rule_sentiment(df_kaggle)
print_confusion(df_kaggle.rule_senti, df_kaggle.sentiment)
#      True    -1     0     1    All
# Predicted                         
# -1         3768   404   241   4413
# 0          3266  2068   997   6331
# 1          2144   627  1125   3896
# All        9178  3099  2363  14640
# Rule Accuracy = 47.53%

###########################################################################################
#### Sentiment Analysis based on RULES using VADER
###########################################################################################
from nltk import tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

#start process_tweet but keep the upper caps this time around
def processTweet_keepCap(tweet):
    # process the tweets

    #Convert to lower case
    #tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub(r'@([^\s]+)', r'\1', tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Replace '&amp;' with 'and'
    tweet = re.sub(r'&amp;', r' and', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

# This function will add a pre-processing text column with not lowering all cases    
def adding_ppcontent_cap_col(df):
    pptext = []
    for line in df.content:
        pptext.append(processTweet_keepCap(line))
    df['ppcontent_cap'] = pptext
# END adding_ppcontent_cap_col()

adding_ppcontent_cap_col(df_text)
adding_ppcontent_cap_col(df_2400)
adding_ppcontent_cap_col(df_nr)
adding_ppcontent_cap_col(df_kaggle)


## This function will add a new column "polarity_sentiment" to the df
def adding_polarity_sentiment_col(df):
    #Must already have the 'ppcontent_cap' column
    polarity = []
    for line in df.ppcontent_cap:
        min_ss = {'compound': 0, 'neg': 1, 'neu': 1, 'pos': 1}
        max_ss = {'compound': 0, 'neg': 1, 'neu': 1, 'pos': 1}
        sentences = tokenize.sent_tokenize(line)
        for sentence in sentences:
            ss = sid.polarity_scores(sentence)
            if ss['compound'] < min_ss['compound']:
                min_ss = ss
            elif ss['compound'] > max_ss['compound']:
                max_ss = ss
        if min_ss == max_ss:
            polarity.append(ss)
        elif min_ss != {'compound': 0, 'neg': 1, 'neu': 1, 'pos': 1}:
            polarity.append(min_ss)
        else:
            polarity.append(max_ss)

    df['polarity'] = polarity
    
    senti2 = []
    for dic in polarity:
        if dic['compound'] == 0:
            senti2.append(0)
        elif dic['compound'] < 0:
            senti2.append(-1)
        else:
            senti2.append(1)

    df['polarity_senti'] = senti2
#End function

################################
## Now let's call these function
## and look at the confusion!
################################
adding_polarity_sentiment_col(df_2400)
print_confusion(df_2400.polarity_senti, df_2400.sentiment)
#      True    -1    0    1   All
# Predicted                      
# -1          768   70   31   869
# 0           195  191   25   411
# 1           470  295  355  1120
# All        1433  556  411  2400
# Rule Accuracy: 54.75%

adding_polarity_sentiment_col(df_text)

adding_polarity_sentiment_col(df_nr)
print_confusion(df_nr.polarity_senti, df_nr.sentiment)
#      True   -1     0    1   All
# Predicted                      
# -1         144   628  117   889
# 0          271   674  130  1075
# 1           78  1256  702  2036
# All        493  2558  949  4000
# Rule Accuracy: 38.00%

adding_polarity_sentiment_col(df_kaggle)
print_confusion(df_kaggle.polarity_senti, df_kaggle.sentiment)
#      True    -1     0     1    All
# Predicted                         
# -1         4890   477   197   5564
# 0          1221  1008   155   2384
# 1          3067  1614  2011   6692
# All        9178  3099  2363  14640
# Rule Accuracy: 54.02%

#################################################################################################
#################################################################################################
#####
##### MACHINE LEARNING Based Sentiment Analysis
##### 
#####            BINARY Classifier  
#####
#################################################################################################
#################################################################################################

#################################################################################################
#### Data Preparation - Split to Training and Testing
####                    Eliminating Neutral Sentiments
#################################################################################################
from sklearn import feature_extraction, ensemble, cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier as RFC

## For the df_2400, we wanted to split the data evenly among the airlines
train_aa, test_aa = train_test_split(df_2400[0:600], test_size = 0.2, random_state = 2016)
train_da, test_da = train_test_split(df_2400[6000:1200], test_size = 0.2, random_state = 2016)
train_sw, test_sw = train_test_split(df_2400[1200:1800], test_size = 0.2, random_state = 2016)
train_ua, test_ua = train_test_split(df_2400[1800:2400], test_size = 0.2, random_state = 2016)
df_2400_train = pd.concat([train_aa, train_da, train_sw, train_ua])
df_2400_train.reset_index(inplace=True)
df_2400_test = pd.concat([test_aa, test_da, test_sw, test_ua])
df_2400_test.reset_index(inplace=True)


## for the df_2400, eliminate all netural rating
df_2400bi_train = df_2400_train[df_2400_train.sentiment != 0].copy()
df_2400bi_test = df_2400_test[df_2400_test.sentiment != 0].copy()
df_2400bi = df_2400[df_2400.sentiment != 0].copy()

## We then do the same for df_kaggle dataset
df_kaggle_train, df_kaggle_test = train_test_split(df_kaggle, test_size = 0.2, random_state = 2016)

df_kagglebi_train = df_kaggle_train[df_kaggle_train.sentiment != 0].copy()
df_kagglebi_test = df_kaggle_test[df_kaggle_test.sentiment != 0].copy()
df_kagglebi = df_kaggle[df_kaggle.sentiment != 0].copy()


## Instantiate a new CountVectorizer with stop words
vectorizer = feature_extraction.text.CountVectorizer(stop_words = 'english')
#vectorizer = feature_extraction.text.TfidfVectorizer(stop_words = 'english')

####################
## Transform df_2400
####################
## Use `fit` to learn the vocabulary of the reviews
vectorizer.fit(df_2400bi_train.ppcontent)
#vectorizer.get_feature_names() #bag-of-words

# Use `tranform` to generate the sample X word matrix - one column per word
X_2400bi_train = vectorizer.transform(df_2400bi_train.ppcontent)
X_2400bi_test = vectorizer.transform(df_2400bi_test.ppcontent)
y_2400bi_train = df_2400bi_train.sentiment
y_2400bi_test = df_2400bi_test.sentiment

X_2400_train = vectorizer.transform(df_2400_train.ppcontent)
X_2400_test = vectorizer.transform(df_2400_test.ppcontent)
y_2400_train = df_2400_train.sentiment
y_2400_test = df_2400_test.sentiment

######################
## Transform df_kaggle
######################
## Use `fit` to learn the vocabulary of the reviews
vectorizer.fit(df_kagglebi_train.ppcontent)
#vectorizer.get_feature_names() #bag-of-words

# Use `tranform` to generate the sample X word matrix - one column per word
X_kagglebi_train = vectorizer.transform(df_kagglebi_train.ppcontent)
X_kagglebi_test = vectorizer.transform(df_kagglebi_test.ppcontent)
y_kagglebi_train = df_kagglebi_train.sentiment
y_kagglebi_test = df_kagglebi_test.sentiment

X_kaggle_train = vectorizer.transform(df_kaggle_train.ppcontent)
X_kaggle_test = vectorizer.transform(df_kaggle_test.ppcontent)
y_kaggle_train = df_kaggle_train.sentiment
y_kaggle_test = df_kaggle_test.sentiment


###############################################
## Combined df_kaggle and df_2400 and transform
###############################################
cols = ['tweet_id', 'airline', 'content', 'ppcontent', 'fcontent', 'ppcontent_cap', 'polarity',
        'rule_senti', 'polarity_senti', 'sentiment']
df_combbi = pd.concat([df_kagglebi[cols], df_2400bi[cols]])
df_comb = pd.concat([df_kaggle[cols], df_2400[cols]])

df_combbi_train = df_combbi[0:11541]
df_combbi_test = df_combbi[11541:13384]

df_comb_train = df_comb[0:14640]
df_comb_test = df_comb[14640:17041]

## Use `fit` to learn the vocabulary of the reviews
vectorizer.fit(df_combbi_train.ppcontent)
#vectorizer.get_feature_names() #bag-of-words

# Use `tranform` to generate the sample X word matrix - one column per word
X_combbi_train = vectorizer.transform(df_combbi_train.ppcontent)
X_combbi_test = vectorizer.transform(df_combbi_test.ppcontent)
y_combbi_train = df_combbi_train.sentiment
y_combbi_test = df_combbi_test.sentiment

X_comb_train = vectorizer.transform(df_comb_train.ppcontent)
X_comb_test = vectorizer.transform(df_comb_test.ppcontent)
y_comb_train = df_comb_train.sentiment
y_comb_test = df_comb_test.sentiment




#################################################################################################
#### RANDOM FOREST - BULIDING THE MODEL
#################################################################################################
## Here's a function that plots the ROC curve as well as report model accuracy
def plot_bi_roc (X_transformed, y, model, title_text):
    y_hat = model.predict(X_transformed)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    plt.figure()
    plt.plot(fpr, tpr, label = 'ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([.0, 1.])
    plt.ylim([.0, 1.1])
    plt.xlabel('FPR/Fall-out')
    plt.ylabel('TPR/Sensitivity')
    plt.title(title_text+' Sentiment ROC')
    plt.legend(loc = 'lower right')
    plt.show()
    print()
    print_confusion(y_hat, y)
    print ('Your Model Score is {:2.4}%'.format(model.score(X_transformed, y)*100))
#END plot_bi_roc ()

## Let's build a model and use a "balanced" class-weight
rfc_model = ensemble.RandomForestClassifier(n_estimators = 100, class_weight='balanced', random_state = 2016)
#cross_validation.cross_val_score(model, train_X_transformed, train_y, scoring = 'roc_auc')


######################
## Model for df_2400bi
######################
rfc_model.fit(X_2400bi_train, y_2400bi_train)

## RFC Results for Training set
plot_bi_roc(X_2400bi_train, y_2400bi_train, rfc_model, 'Training')
## RFC Results for Testing set
plot_bi_roc(X_2400bi_test, y_2400bi_test, rfc_model, 'Testing')

#      True   -1   1  All
# Predicted              
# -1         213  33  246
# 1           12  32   44
# All        225  65  290
# Your Model Score is 84.48%

########################
## Model for df_kagglebi
########################
rfc_model.fit(X_kagglebi_train, y_kagglebi_train)

## RFC Results for Training set
plot_bi_roc(X_kagglebi_train, y_kagglebi_train, rfc_model, 'Training')
## RFC Results for Testing set
plot_bi_roc(X_kagglebi_test, y_kagglebi_test, rfc_model, 'Testing')

#      True    -1    1   All
# Predicted                 
# -1         1725  169  1894
# 1            68  313   381
# All        1793  482  2275
# Your Model Score is 89.58%

######################
## Model for df_combbi
######################
rfc_model.fit(X_combbi_train, y_combbi_train)

## RFC Results for Training set
plot_bi_roc(X_combbi_train, y_combbi_train, rfc_model, 'Training')
## RFC Results for Testing set
plot_bi_roc(X_combbi_test, y_combbi_test, rfc_model, 'Testing')

#      True    -1    1   All
# Predicted                 
# -1         1334  150  1484
# 1            99  260   359
# All        1433  410  1843
# Your Model Score is 86.49%



#################################################################################################
#### SUPPORT VECTOR MACHINE - BULIDING THE MODEL
#################################################################################################
from sklearn import svm
svc_model = svm.LinearSVC(penalty = 'l1', dual=False, C=1.0, random_state=2016)

######################
## Model for df_2400bi
######################
svc_model.fit(X_2400bi_train, y_2400bi_train)

plot_bi_roc(X_2400bi_train, y_2400bi_train, svc_model, 'Training')
plot_bi_roc(X_2400bi_test, y_2400bi_test, svc_model, 'Testing')

#      True   -1   1  All
# Predicted              
# -1         208  29  237
# 1           17  36   53
# All        225  65  290
# Your Model Score is 84.14%

########################
## Model for df_kagglebi
########################
svc_model.fit(X_kagglebi_train, y_kagglebi_train)

plot_bi_roc(X_kagglebi_train, y_kagglebi_train, svc_model, 'Training')
plot_bi_roc(X_kagglebi_test, y_kagglebi_test, svc_model, 'Testing')

#      True    -1    1   All
# Predicted                 
# -1         1714  127  1841
# 1            79  355   434
# All        1793  482  2275
# Your Model Score is 90.95%

######################
## Model for df_combbi
######################
svc_model.fit(X_combbi_train, y_combbi_train)

plot_bi_roc(X_combbi_train, y_combbi_train, svc_model, 'Training')
plot_bi_roc(X_combbi_test, y_combbi_test, svc_model, 'Testing')

#      True    -1    1   All
# Predicted                 
# -1         1317  135  1452
# 1           116  275   391
# All        1433  410  1843
# Your Model Score is 86.38%



#################################################################################################
#################################################################################################
#####
##### MACHINE LEARNING Based Sentiment Analysis
#####           Muti-Class Classifier   
#####  Using Entire Kaggle Dataset as Training
#####
#################################################################################################
#################################################################################################

#################################################################################################
#### LINEAR SVM (or 'SVC') MODEL Parameters Selection
####                     Part 1
#### Using only 'df_2400_train' to Train and Validate 
#################################################################################################
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
from scipy.stats.stats import pearsonr
import itertools

'''
from sklearn import grid_search

parameters = {'decision_function_shape':('ovo', 'ovr'), 'C':[0.01, 0.1, 1, 10, 100]}
svr = svm.SVC(kernel = 'linear', random_state = 2016, gamma = 0)
clf = grid_search.GridSearchCV(svr, parameters, cv=5, n_jobs=4)
clf.fit(X_2400_train, y_2400_train)
print(clf.best_params_)
# {'decision_function_shape': 'ovo', 'C': 0.1}

parameters = {'decision_function_shape':('ovo'), 'C':[0.05, 0.075, 0.1, 0.2, 0.4]}
svr = svm.SVC(kernel = 'linear', random_state = 2016, gamma = 0)
clf = grid_search.GridSearchCV(svr, parameters, cv=5, n_jobs=4)
clf.fit(X_2400_train, y_2400_train)
print(clf.best_params_)
# {'decision_function_shape': 'ovo', 'C': 0.2}

parameters = {'C':[0.15, 0.175, 0.2, 0.25, 0.3]}
svr = svm.SVC(decision_function_shape = 'ovo', kernel = 'linear', random_state = 2016, gamma = 0)
clf = grid_search.GridSearchCV(svr, parameters, cv=5, n_jobs=4)
clf.fit(X_2400_train, y_2400_train)
print(clf.best_params_)
{'C': 0.15}
'''

svm_model2 = svm.SVC(decision_function_shape='ovo', kernel = 'linear', C=0.15, gamma=0, random_state=2016)

skf2 = StratifiedKFold(y_2400_train, n_folds=5, random_state=2016)
accu_cv2 = []
true_cv = []
pred_cv = []
for train_index, valid_index in skf2:
    svm_model2.fit(X_2400_train[train_index], y_2400_train[train_index])
    y_hat2 = svm_model2.predict(X_2400_train[valid_index])
    print_confusion(y_hat2, y_2400_train[valid_index])
    accuracy = svm_model2.score(X_2400_train[valid_index], y_2400_train[valid_index])
    print ('Your Model Score is {:2.4}%'.format(accuracy*100))
    accu_cv2.append(accuracy)
    true_cv.append(y_comb_train[valid_index])
    pred_cv.append(y_hat2)
print('')
print('The Average Score is {:2.4}%'.format(np.average(accu_cv2)*100))
# Model Score CV.1: 67.13%
# Model Score CV.2: 66.32%
# Model Score CV.3: 59.72%
# Model Score CV.4: 70.83%
# Model Score CV.5: 67.94%
# The Average Score is: 66.39%

## Here we Correlation between the truth and prediction
TrueLabel = list(itertools.chain(*true_cv))
PredictedLabel = list(itertools.chain(*pred_cv))
print ('Correlation between the actual and prediction is:', pearsonr(TrueLabel, PredictedLabel)[0], \
       'with p-value',  ("%2.2f" % pearsonr(TrueLabel, PredictedLabel)[1]))

## Here we plot out the confusion matrix of the Cross-Validation Results
cm = confusion_matrix(PredictedLabel, TrueLabel)
fig, ax = plt.subplots()
im = ax.matshow(cm)
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
plt.title('Confusion matrix')
fig.colorbar(im)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#################################################################################################
#### Testing the model result on 'df_2400_test', a subset of df_2400
#################################################################################################
## Here's a function that reports model accuracy
def multi_class_outcome (X_transformed, y, model):
    y_hat = model.predict(X_transformed)
    print_confusion(y_hat, y)
    print ('Your Model Score is {:2.4}%'.format(model.score(X_transformed, y)*100))
#END multi_class_model_outcome ()
    
svm_model2.fit(X_2400_train, y_2400_train)
multi_class_outcome(X_2400_train, y_2400_train, svm_model2)
multi_class_outcome(X_2400_test, y_2400_test, svm_model2)
#      True   -1   0   1  All
# Predicted                  
# -1         210  57  32  299
# 0            6  10   2   18
# 1            9   3  31   43
# All        225  70  65  360
# Your Model Score is 69.72%


## Here we Correlation between the truth and prediction
TrueLabel = list(itertools.chain(*true_cv))
PredictedLabel = list(itertools.chain(*pred_cv))
print ('Correlation between the actual and prediction is:', pearsonr(TrueLabel, PredictedLabel)[0], \
       'with p-value',  ("%2.2f" % pearsonr(TrueLabel, PredictedLabel)[1]))

## Here we plot out the confusion matrix of the Cross-Validation Results
cm = confusion_matrix(PredictedLabel, TrueLabel)
fig, ax = plt.subplots()
im = ax.matshow(cm)
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
plt.title('Confusion matrix')
fig.colorbar(im)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#################################################################################################
#### LINEAR SVM (or 'SVC') MODEL Parameters Selection
####                   Part 2
#### Using 'df_comb_train' to Train and Validate 
#################################################################################################
'''
parameters = {'decision_function_shape':('ovo', 'ovr'), 'C':[0.01, 0.1, 1, 10, 100]}
svr = svm.SVC(kernel = 'linear', random_state = 2016)
clf = grid_search.GridSearchCV(svr, parameters, cv=5, n_jobs=4)
clf.fit(X_comb_train, y_comb_train)
print(clf.best_params_)
# {'decision_function_shape': 'ovo', 'C': 1}

parameters = {'gamma':[0.01, 0.1, 1, 10, 100], 'C':[0.5, 0.75, 1, 2.5, 5]}
svr = svm.SVC(kernel = 'linear', random_state = 2016)
clf = grid_search.GridSearchCV(svr, parameters, cv=5, n_jobs=4)
clf.fit(X_comb_train, y_comb_train)
print(clf.best_params_)
# {'gamma': 0.01, 'C': 0.5}

parameters = {'gamma':[0.0, 0.05, 0.01, 0.05, 0.10], 'C':[0.3, 0.4, 0.5, 0.6, 0.7]}
svr = svm.SVC(kernel = 'linear', decision_function_shape ='ovo', random_state = 2016)
clf = grid_search.GridSearchCV(svr, parameters, cv=5, n_jobs=4)
clf.fit(X_comb_train, y_comb_train)
print(clf.best_params_)
# {'gamma': 0.0, 'C': 0.4}
'''

## Verifying result using Cross Validation on the entire Data Set: n_fold=5
skf = StratifiedKFold(y_comb_train, n_folds=5, random_state=2016)

svm_model = svm.SVC(decision_function_shape='ovo', kernel='linear', C=0.4, gamma=0, random_state=2016)
accu_cv = []
true_cv = []
pred_cv = []
for train_index, valid_index in skf:
    svm_model.fit(X_comb_train[train_index], y_comb_train[train_index])
    y_hat = svm_model.predict(X_comb_train[valid_index])
    print_confusion(y_hat, y_comb_train[valid_index])
    accuracy = svm_model.score(X_comb_train[valid_index], y_comb_train[valid_index])
    print ('Your Model Score is {:2.4}%'.format(accuracy*100))
    accu_cv.append(accuracy)
    true_cv.append(y_comb_train[valid_index])
    pred_cv.append(y_hat)
print('')
print('The Average Score is {:2.4}%'.format(np.average(accu_cv)*100))

# Model Score CV.1: 75.66%
# Model Score CV.2: 75.15%
# Model Score CV.3: 58.79%
# Model Score CV.4: 75.78%
# Model Score CV.5: 74.16%
# The Average Score is: 71.91%

## Here we Correlation between the truth and prediction
TrueLabel = list(itertools.chain(*true_cv))
PredictedLabel = list(itertools.chain(*pred_cv))
print ('Correlation between the actual and prediction is:', pearsonr(TrueLabel, PredictedLabel)[0], \
       'with p-value',  ("%2.2f" % pearsonr(TrueLabel, PredictedLabel)[1]))

## Here we plot out the confusion matrix of the Cross-Validation Results
cm = confusion_matrix(PredictedLabel, TrueLabel)
fig, ax = plt.subplots()
im = ax.matshow(cm)
for (i, j), z in np.ndenumerate(cm):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
plt.title('Confusion matrix')
fig.colorbar(im)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#################################################################################################
#### Testing the model result on 'df_comb_test', a subset of df_comb
#################################################################################################

#svm_model = svm.LinearSVC(penalty = 'l1', dual=False, C=1.0, random_state=2016)
svm_model = svm.SVC(decision_function_shape='ovo', kernel = 'linear', C=0.4, gamma=0, random_state=2016)
svm_model.fit(X_comb_train, y_comb_train)

multi_class_outcome(X_comb_train, y_comb_train, svm_model)
multi_class_outcome(X_comb_test, y_comb_test, svm_model)

#      True    -1    0    1   All
# Predicted                      
# -1         1137  253   97  1487
# 0           217  273   66   556
# 1            79   30  248   357
# All        1433  556  411  2400
# Your Model Score is 69.08%

#################################################################################################
##   CONCLUSION:
##   Given that the test results are very similar - 70.0% vs. 69.08%, we opt to use df_comb
##   to train the model because its test result is still superior to the average cross-validation
##   validation results we got from using df_2400 dataset.  Also, its difference to the average 
##   CV result is also smaller than the df_2400.  Finally with larger training set and larger
##   testing datasets, we are more confident with the results we are seeing.
#################################################################################################



#################################################################################################
#### ADABOOST (DECISION-TREE BASED) MODEL Parameters Selection
####                     Part 1
#### Using only 'df_2400_train' to Train and Validate 
#################################################################################################
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
'''
parameters = {'learning_rate':[0.1, 0.5, 1, 1.5, 2], 'algorithm': ['SAMME', 'SAMME.R']}
ada = AdaBoostClassifier(n_estimators=100, random_state=2016)
clf = grid_search.GridSearchCV(ada, parameters, cv=5, n_jobs=4)
clf.fit(X_2400_train, y_2400_train)
print(clf.best_params_)
# {'learning_rate': 1.5, 'algorithm': 'SAMME'}
'''

ada_model2 = AdaBoostClassifier(n_estimators=100, learning_rate=1.5, algorithm='SAMME', random_state=2016)

skf2 = StratifiedKFold(y_2400_train, n_folds=5, random_state=2016)
accu_cv2 = []
true_cv = []
pred_cv = []
for train_index, valid_index in skf2:
    ada_model2.fit(X_2400_train[train_index], y_2400_train[train_index])
    y_hat2 = ada_model2.predict(X_2400_train[valid_index])
    print_confusion(y_hat2, y_2400_train[valid_index])
    accuracy = ada_model2.score(X_2400_train[valid_index], y_2400_train[valid_index])
    print ('Your Model Score is {:2.4}%'.format(accuracy*100))
    accu_cv2.append(accuracy)
    true_cv.append(y_comb_train[valid_index])
    pred_cv.append(y_hat2)
print('')
print('The Average Score is {:2.4}%'.format(np.average(accu_cv2)*100))
# Model Score CV.1: 63.32%
# Model Score CV.2: 68.40%
# Model Score CV.3: 58.68%
# Model Score CV.4: 69.10%
# Model Score CV.5: 64.11%
# The Average Score is: 64.86%

ada_scores = cross_val_score(ada_model2, X_2400_train, y_2400_train)
print(ada_scores.mean())
#53.31%

#################################################################################################
#### Testing the model result on 'df_2400_test', a subset of df_2400
#################################################################################################
ada_scores = cross_val_score(ada_model2, X_2400_train, y_2400_train)
print(ada_scores.mean())
#53.31%

ada_model2.fit(X_2400_train, y_2400_train)
multi_class_outcome(X_2400_train, y_2400_train, ada_model2)
multi_class_outcome(X_2400_test, y_2400_test, ada_model2)
#      True   -1   0   1  All
# Predicted                  
# -1         218  69  42  329
# 0            1   1   1    3
# 1            6   0  22   28
# All        225  70  65  360
# Your Model Score is 66.94%


#################################################################################################
#### ADABOOS (DECISION-TREE BASED) MODEL Parameters Selection
####                   Part 2
#### Using 'df_comb_train' to Train and Validate 
#################################################################################################
'''
parameters = {'learning_rate':[0.1, 0.5, 1, 1.5, 2], 'algorithm': ['SAMME', 'SAMME.R']}
ada = AdaBoostClassifier(n_estimators=100, random_state=2016)
clf = grid_search.GridSearchCV(ada, parameters, cv=5, n_jobs=4)
clf.fit(X_comb_train, y_comb_train)
print(clf.best_params_)
#
'''

## Verifying result using Cross Validation on the entire Data Set: n_fold=5
skf = StratifiedKFold(y_comb_train, n_folds=5, random_state=2016)

ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=1.5, algorithm='SAMME.R', random_state=2016)
accu_cv = []
true_cv = []
pred_cv = []
for train_index, valid_index in skf:
    ada_model.fit(X_comb_train[train_index], y_comb_train[train_index])
    y_hat = ada_model.predict(X_comb_train[valid_index])
    print_confusion(y_hat, y_comb_train[valid_index])
    accuracy = ada_model.score(X_comb_train[valid_index], y_comb_train[valid_index])
    print ('Your Model Score is {:2.4}%'.format(accuracy*100))
    accu_cv.append(accuracy)
    true_cv.append(y_comb_train[valid_index])
    pred_cv.append(y_hat)
print('')
print('The Average Score is {:2.4}%'.format(np.average(accu_cv)*100))
# Model Score CV.1: 70.88%
# Model Score CV.2: 68.45%
# Model Score CV.3: 56.20%
# Model Score CV.4: 73.15%
# Model Score CV.5: 70.33%
# The Average Score is: 67.80%


#################################################################################################
#### Testing the model result on 'df_comb_test', a subset of df_comb
#################################################################################################
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=1.5, algorithm='SAMME.R', random_state=2016)

ada_scores = cross_val_score(ada_model, X_comb_train, y_comb_train)
print(ada_scores.mean())
#63.27%

ada_model.fit(X_comb_train, y_comb_train)
multi_class_outcome(X_comb_train, y_comb_train, ada_model)
multi_class_outcome(X_comb_test, y_comb_test, ada_model)
#      True    -1    0    1   All
# Predicted                      
# -1          988  259   82  1329
# 0           329  252   66   647
# 1           116   45  263   424
# All        1433  556  411  2400
# Your Model Score is 62.62%

#################################################################################################
##   CONCLUSION:
##   AdaBoostClassifier isn't superior to Linear SVM (SVC) Model above in both training methods.
#################################################################################################


print('The Average Score is {:2.4}%'.format(np.average(accu_cv2)*100))
multi_class_outcome(X_2400_test, y_2400_test, ada_model2)

print('The Average Score is {:2.4}%'.format(np.average(accu_cv)*100))
multi_class_outcome(X_comb_test, y_comb_test, ada_model)



'''
parameters = {'learning_rate':[0.1, 0.5, 1, 1.5, 2], 'algorithm': ['SAMME', 'SAMME.R']}
ada = AdaBoostClassifier(n_estimators=100, random_state=2016)
clf = grid_search.GridSearchCV(ada, parameters, cv=5, n_jobs=4)
clf.fit(X_comb_train, y_comb_train)
print(clf.best_params_)
#
'''

## Verifying result using Cross Validation on the entire Data Set: n_fold=5
skf = StratifiedKFold(y_comb_train, n_folds=5, random_state=2016)

ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=1.5, algorithm='SAMME.R', random_state=2016)
accu_cv = []
true_cv = []
pred_cv = []
for train_index, valid_index in skf:
    ada_model.fit(X_comb_train[train_index], y_comb_train[train_index])
    y_hat = ada_model.predict(X_comb_train[valid_index])
    print_confusion(y_hat, y_comb_train[valid_index])
    accuracy = ada_model.score(X_comb_train[valid_index], y_comb_train[valid_index])
    print ('Your Model Score is {:2.4}%'.format(accuracy*100))
    accu_cv.append(accuracy)
    true_cv.append(y_comb_train[valid_index])
    pred_cv.append(y_hat)
print('')
print('The Average Score is {:2.4}%'.format(np.average(accu_cv)*100))

'''
#################################################################################################
#### Gradient Boosting Classifier - Trying another models to assess different results
#### While it seems superior to AdaBoostClassifer by just a little bit, the algorithm performs
#### way too slowly for me to be productive.
#################################################################################################
from sklearn.ensemble import GradientBoostingClassifier

gra_model = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0, max_depth=1, presort=False, random_state=2016)
gra_scores = cross_val_score(gra_model, X_comb_train.toarray(), y_comb_train)
print(gra_scores.mean())
#0.664958839611

gra_model.fit(X_comb_train.toarray(), y_comb_train)
multi_class_outcome(X_comb_train, y_comb_train, ada_model)
multi_class_outcome(X_comb_test, y_comb_test, ada_model)
'''



#################################################################################################
#################################################################################################
#####
##### 
##### PREDICTING SENTIMENT OF THE UNKONWN DATASETS USING OUR BEST MODEL!!
#####  
#####
#################################################################################################
#################################################################################################

###############################################
## Combined df_kaggle and df_text and transform
###############################################
cols = ['tweet_id', 'airline', 'content', 'ppcontent', 'fcontent', 'ppcontent_cap', 'polarity',
        'rule_senti', 'polarity_senti', 'retweet', 'sentiment',
        'coordinates', 'media_type', 'video_link', 'photo_link', 'twitpic', 'source']
        
df_kaggle['retweet'] = 'no'
df_kaggle['coordinates'] = np.nan
df_kaggle['media_type'] = np.nan
df_kaggle['video_link'] = np.nan
df_kaggle['photo_link'] = np.nan
df_kaggle['twitpic'] = np.nan
df_kaggle['source'] = np.nan
df_text['sentiment'] = np.nan
df_full = pd.concat([df_kaggle[cols], df_text[cols]])

df_full_train = df_full[0:14640].copy()    #Training set is essentially the Kaggle Dataset
df_full_new = df_full[14640:94761].copy()  #This is the unknown dataset

## Use `fit` to learn the vocabulary of the reviews
vectorizer.fit(df_full_train.ppcontent)
#vectorizer.get_feature_names() #bag-of-words

# Use `tranform` to generate the sample X word matrix - one column per word
X_full_train = vectorizer.transform(df_full_train.ppcontent)
X_full_new = vectorizer.transform(df_full_new.ppcontent)
y_full_train = df_full_train.sentiment
#y_full_new = df_full_test.sentiment #this line is useless due to all null values

## Our Best Model
svm_model = svm.SVC(decision_function_shape='ovo', kernel = 'linear', C=0.4, gamma=0, random_state=2016)
svm_model.fit(X_full_train, y_full_train)

y_hat = svm_model.predict(X_full_new)
df_full_new.sentiment = y_hat


## We replaced the predicted sentiment score with the manual graded scores from the 2,400 subset
df_truth = pd.read_csv('C:/Users/Michael Lin_2/Desktop/Airline Databases/truth_combined.csv', encoding='latin-1')
df_truth.sentiment = df_truth.sentiment - 2
x1 = df_full_new.set_index(['tweet_id'])['sentiment']
x2 = df_truth.set_index(['tweet_id'])['sentiment']
x1.update(x2)                           # call update function, this is inplace
df_full_new['sentiment'] = x1.values    # replace the values in original df1

# Double check results
df_truth.sort(['tweet_id'], ascending=[True], inplace=True)
(df_full_new.tweet_id.isin(df_truth.tweet_id)).value_counts()
np.bincount(x1.index.isin(x2.index))



#################################################################################################
#################################################################################################
#####
##### 
##### STATISTICS AND RESULTS
#####  
#####
#################################################################################################
#################################################################################################

######################################
## Number of Unique Tweets Per Airline
######################################
df_full_new.groupby(['airline']).retweet.value_counts()
df_full_new.groupby(['airline', 'retweet']).sentiment.value_counts()
df_full_new.groupby(['retweet']).sentiment.value_counts()

## Coordinate turned out to be useless
coord = df_full_new.coordinates.dropna()


df_full_new[df_full_new.sentiment == 1].source.value_counts(normalize=True).head(10)
df_full_new[df_full_new.sentiment == 0].source.value_counts(normalize=True).head(10)
df_full_new[df_full_new.sentiment == -1].source.value_counts(normalize=True).head(10)


df_full_new[df_full_new.photo_link == 1].sentiment.value_counts()
df_full_new[df_full_new.video_link == 1].sentiment.value_counts()
df_full_new.groupby(['media_type']).sentiment.value_counts()

##################################################
## Top 200 negative words for AA (Original Tweets)
##################################################
from collections import Counter

## This function will return the top 200 frequent words in dictionary
def top_words_dict(text):
    # Pass a Series of list of strings
    word_dic = {}
    for i in text:
        line = Counter(i)
        word_dic = {k: word_dic.get(k, 0) + line.get(k, 0) for k in set(word_dic) | set(line)}
    word_200 = dict(Counter(word_dic).most_common(300))
    return word_200
# END of top_words_dict()

row_index = (df_full_new.sentiment == -1) & (df_full_new.airline == 'AA') & (df_full_new.retweet == 'no')
text_neg_AA = df_full_new.fcontent[row_index].values.tolist()
top_neg_AA = top_words_dict(text_neg_AA)

##################################################
## Top 200 positive words for AA (Original Tweets)
##################################################
row_index = (df_full_new.sentiment == 1) & (df_full_new.airline == 'AA') & (df_full_new.retweet == 'no')
text_pos_AA = df_full_new.fcontent[row_index].values.tolist()
top_pos_AA = top_words_dict(text_pos_AA)



##################################################
## Top 200 negative words for DA (Original Tweets)
##################################################
row_index = (df_full_new.sentiment == -1) & (df_full_new.airline == 'DA') & (df_full_new.retweet == 'no')
text_neg_DA = df_full_new.fcontent[row_index].values.tolist()
top_neg_DA = top_words_dict(text_neg_DA)

##################################################
## Top 200 positive words for DA (Original Tweets)
##################################################
row_index = (df_full_new.sentiment == 1) & (df_full_new.airline == 'DA') & (df_full_new.retweet == 'no')
text_pos_DA = df_full_new.fcontent[row_index].values.tolist()
top_pos_DA = top_words_dict(text_pos_DA)



##################################################
## Top 200 negative words for SW (Original Tweets)
##################################################
row_index = (df_full_new.sentiment == -1) & (df_full_new.airline == 'SW') & (df_full_new.retweet == 'no')
text_neg_SW = df_full_new.fcontent[row_index].values.tolist()
top_neg_SW = top_words_dict(text_neg_SW)

##################################################
## Top 200 positive words for SW (Original Tweets)
##################################################
row_index = (df_full_new.sentiment == 1) & (df_full_new.airline == 'SW') & (df_full_new.retweet == 'no')
text_pos_SW = df_full_new.fcontent[row_index].values.tolist()
top_pos_SW = top_words_dict(text_pos_SW)



##################################################
## Top 200 negative words for UA (Original Tweets)
##################################################
row_index = (df_full_new.sentiment == -1) & (df_full_new.airline == 'UA') & (df_full_new.retweet == 'no')
text_neg_UA = df_full_new.fcontent[row_index].values.tolist()
top_neg_UA = top_words_dict(text_neg_UA)

##################################################
## Top 200 positive words for UA (Original Tweets)
##################################################
row_index = (df_full_new.sentiment == 1) & (df_full_new.airline == 'UA') & (df_full_new.retweet == 'no')
text_pos_UA = df_full_new.fcontent[row_index].values.tolist()
top_pos_UA = top_words_dict(text_pos_UA)


with open('top_neg_AA.csv', 'w', encoding ='utf-8') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in top_neg_AA.items()]
with open('top_pos_AA.csv', 'w', encoding ='utf-8') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in top_pos_AA.items()]

with open('top_neg_DA.csv', 'w', encoding ='utf-8') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in top_neg_DA.items()]
with open('top_pos_DA.csv', 'w', encoding ='utf-8') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in top_pos_DA.items()]

with open('top_neg_SW.csv', 'w', encoding ='utf-8') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in top_neg_SW.items()]
with open('top_pos_SW.csv', 'w', encoding ='utf-8') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in top_pos_SW.items()]

with open('top_neg_UA.csv', 'w', encoding ='utf-8') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in top_neg_UA.items()]
with open('top_pos_UA.csv', 'w', encoding ='utf-8') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in top_pos_UA.items()]
    
##################################################
## Top 200 positive words for UA (Original Tweets)
##################################################
more_col = ['coordinates', 'media_type', 'video_link', 'photo_link', 'twitpic', 'source']

df_full_new['coordinates'] = np.nan
df_full_new['media_type'] = np.nan
df_full_new['video_link'] = np.nan
