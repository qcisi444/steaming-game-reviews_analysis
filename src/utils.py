import jsonlines
import pandas as pd
import numpy as np

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from many_stop_words import get_stop_words

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('punkt')

# punctuation includes the symbols '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
punctuation = string.punctuation
# stopwords are words of lesser interesst like 'and, for, no, because, nobody,...'
stopwords = get_stop_words('en')#list(STOP_WORDS)

special_characters = list(string.punctuation)
lemmatizer = WordNetLemmatizer()

# convert data for the given 'keys' from jasonlines file to pandas dataframe
def jasonlines_to_panda_df(path, keys, drop_review_dupes=True):
    feature_dict = {}
    for key in keys:
        feature_dict.update({key:[]})
        
    with jsonlines.open(path) as reader:
        for obj in reader:
            for key in feature_dict.keys():
                value = obj[key]
                if key=='rating':
                    feature_dict[key].append(int(value=='Recommended'))
                elif key =='review':
                    feature_dict[key].append(value)
                elif isinstance(value,str):
                    # default action for other string values with yet undefined purpose (e.g.date)
                    feature_dict[key].append(len(value)) 
                else:
                    feature_dict[key].append(value)    
            
    df = pd.DataFrame(data=feature_dict)
    df.drop_duplicates(subset=['review'], keep='first', inplace=True)
    # Replace NaN with 0
    df.fillna(0,inplace=True)
    return df


# sample the 'review' and 'rating' columns from a game dataset saved as jasonlines file    
def generate_sample_df(game=None, fraction=0.05, number=None, path='../dat'):
    keys = ['review','rating']
    if game is not None:
        df = jasonlines_to_panda_df((path + '/' + game + '.jsonlines'), keys)
    else:
        dfs = {}
        for file in glob.glob(path + "/*.jsonlines"):
            game_df = jasonlines_to_panda_df(file, keys)
            dfs.update({file:game_df})
        df = pd.concat(dfs.values())
    
    n = len(df.index)
    pos = len(df[df['rating']==1].index)
    neg = len(df[df['rating']==0].index)
    assert (pos + neg == n)
    
    if number is not None:
        df_sample = df.sample(n=number, replace=False, random_state=1)
    else:
        df_sample = df.sample(frac=fraction, replace=False, random_state=1)
    
    choice = len(df_sample.index)
    return df_sample, choice, n, pos, neg


# sample from the Fallout_4.csv file in the 'additional_games' data folder    
def get_fallout_4_df(fraction=1, number=None, game='Fallout 4', path='../dat/additional_games'):
    df = pd.read_csv(path+'/Fallout_4.csv')
    n = len(df.index)
    pos = len(df[df['rating']==1].index)
    neg = len(df[df['rating']==0].index)
    if number is not None:
        df_sample = df.sample(n=number, replace=False, random_state=1)
    else:
        df_sample = df.sample(frac=fraction, replace=False, random_state=1)
    choice = len(df_sample.index)
    return df_sample, choice, n, pos, neg


# lemmatize, filter out punctuation and stopwords 
def filter_string(review):
    words = review.split()
    filtered_words = [word for word in words if word not in stopwords]
    filtered_string = ' '.join(filtered_words)
    words = word_tokenize(filtered_string)
    filtered_words = [lemmatizer.lemmatize(word, pos='a') for word in words if word not in stopwords]
    filtered_words = [lemmatizer.lemmatize(word, pos='v') for word in filtered_words if word not in stopwords]
    filtered_words = [lemmatizer.lemmatize(word, pos='n') for word in filtered_words if word not in stopwords]
    filtered_string = ' '.join(filtered_words)
    filtered_string = ''.join(filter(lambda char: char not in special_characters, filtered_string))
    return filtered_string   

# review to lowercase, filter review, sort by rating and re-index
def preprocess_df(df):
    df['review'] = df['review'].apply(lambda x: filter_string(x.lower()))
    # remove empty columns
    df2 = df[df['review'] != '']
    df2 = df2.sort_values(by='rating')
    df2.index = np.arange(0, len(df2))
    return df2


# analyze review texts using the CountVectorizer from the spacy module
def analyze_reviews(df, min_df=20, max_df=0.7, ngram_range=(1, 1)):
    countvectorizer = CountVectorizer(ngram_range=ngram_range, analyzer='word', min_df=min_df, max_df=max_df)
    tfidfvectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer='word', min_df=min_df, max_df=max_df)
    count_wm = countvectorizer.fit_transform(df['review'])
    tfidf_wm = tfidfvectorizer.fit_transform(df['review'])

    count_tokens = countvectorizer.get_feature_names_out()
    tfidf_tokens = tfidfvectorizer.get_feature_names_out()
    
    arrays = [df['rating'], df.index]
    index = pd.MultiIndex.from_arrays(arrays, names=["rating", "id"])

    df_countvec = pd.DataFrame(data=count_wm.toarray(),index=index, columns=count_tokens)
    df_tfidfvec = pd.DataFrame(data=tfidf_wm.toarray(),index=index, columns=tfidf_tokens)

    return df_countvec, df_tfidfvec

# get the total count of a keyword / sequence of keywords appearing and 
#  calculate the relative difference between the appearance in positive and negative reviews, 
#  as well as the relative appearance compared to all reviews
def filter_peaks(df, thresh_rel_diff=0.35):
    num_reviews = len(df.index)
    if not df.index.isin([0],level='rating').any():
        sum_df = df.sum(axis=0)
        sum_df.rename('pos', inplace=True)
        sum_df = sum_df.to_frame()
        sum_df['neg'] = 0
    elif not df.index.isin([1],level='rating').any():
        sum_df = df.sum(axis=0)
        sum_df.rename('neg', inplace=True)
        sum_df = sum_df.to_frame()
        sum_df['pos'] = 0
    else:
        sum_df_neg = df.loc[0].sum(axis=0)
        sum_df_neg.rename('neg', inplace=True)
        sum_df_pos = df.loc[1].sum(axis=0)
        sum_df_pos.rename('pos', inplace=True)
        sum_df = pd.concat([sum_df_neg, sum_df_pos],axis=1)

    sum_df['rel_diff'] = abs(sum_df['neg']-sum_df['pos']) / (sum_df['neg']+sum_df['pos'])
    sum_df['neg_rel_app'] = sum_df['neg']/num_reviews
    sum_df['pos_rel_app'] = sum_df['pos']/num_reviews
    
    sum_df.drop(sum_df[sum_df['rel_diff'] < thresh_rel_diff].index, inplace=True)
    
    return sum_df
