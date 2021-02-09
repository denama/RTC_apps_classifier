#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from config import config_dict

from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
import numpy as np
#from nltk.tokenize import word_tokenize


class DataVectorizer():

    def __init__ (self, dff, conf_dict):

        self.dff = dff
        self.app_list = config_dict["app_list"]
#        self.feature_column = config_dict["domain_level"]
#        self.min_df = config_dict["vectorizer_min_df"]
#        self.max_df = config_dict["vectorizer_max_df"]
        self.feature_column = conf_dict["domain_level"]
        self.min_df = conf_dict["vectorizer_min_df"]
        self.max_df = conf_dict["vectorizer_max_df"]

        self.labels_numeric = {name: i for i, name in enumerate(self.app_list)}




    def get_labels_numeric(self):
        return self.labels_numeric


    def make_df_classify(self):

        #print("\nLabels: ", self.labels_numeric)
        self.df_classify = self.dff[["description", "second_level_domains", "label", "person"]] \
                                    [self.dff["label"].isin(self.app_list)] \
                                    .copy()
        self.df_classify["description"] = self.df_classify["description"].apply(lambda y: np.nan if len(y)==0 else y)
        self.df_classify["second_level_domains"] = self.df_classify["second_level_domains"].apply(lambda y: np.nan if len(y)==0 else y)
        self.df_classify.dropna(inplace=True)

        self.class_samples = self.df_classify["label"].value_counts()
        #print("\nClass samples:\n", self.class_samples)
        self.df_classify["label"] = self.df_classify["label"].map(lambda x: self.labels_numeric[x])
        self.df_classify["description"] = self.df_classify["description"].apply(lambda x: " ".join(x))
        self.df_classify["second_level_domains"] = self.df_classify["second_level_domains"].apply(lambda x: " ".join(x))

        return self.df_classify



    def create_Xy(self):

        self.vectorizer = TfidfVectorizer()

        #Word tokenize would give back all the domains
        #vectorizer.set_params(tokenizer=word_tokenize)

        #Token pattern - regex to follow to find tokens
        self.vectorizer.set_params(token_pattern = "(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]")

        #vectorizer.set_params(token_pattern = '(?u)\\b\\w*[a-zA-Z]\\w+\\b')
        #vectorizer.set_params(token_pattern = '(?![\d]+)(?:[a-zA-Z0-9\.]+)(?![\d]+)')
        #Remove pure number features: '(?u)\\b\\w*[a-zA-Z]\\w*\\b'
        #Remove tokens that contain one number/letter w+ instead of w*

        # include 1-grams and 2-grams - grams are words in our case
        #vectorizer.set_params(ngram_range=(1, 2))

        #Words to remove from vocabulary
        #vectorizer.set_params(stop_words = ["www", "youtube", "dropbox", "twitter", "studenti", "polito"])

        # only keep terms that appear in at least min_df documents (if integer), proportion of documents if float
        self.vectorizer.set_params(min_df = self.min_df)

        # ignore terms that appear in more than 50% of the documents
        self.vectorizer.set_params(max_df = self.max_df)

        #Enable inverse-document-frequency reweighting, default True
        #self.vectorizer.set_params(max_features = 100)

        self.X = self.vectorizer.fit_transform(self.df_classify[self.feature_column])
        self.y = self.df_classify["label"]
        self.names = self.df_classify["person"]


        #Explore attributes of vectorizer
        self.feature_names = self.vectorizer.get_feature_names()

        #A mapping of terms to feature indices
        self.vocab = self.vectorizer.vocabulary_


        #Other way
        #len(np.where(X.toarray()[:, vectorizer.vocabulary_.get('0.client-channel.google.com')] != 0)[0])

        # print idf values
        self.df_idf = DataFrame(self.vectorizer.idf_,
                                   index=self.vectorizer.get_feature_names(),
                                   columns=["idf_weights"]).sort_values(by=['idf_weights'],
                                   ascending=False)



        #Terms that were ignored because they either:
        #occurred in too many documents (max_df)
        #occurred in too few documents (min_df)
        #were cut off by feature selection (max_features).

        self.stopwords = self.vectorizer.stop_words_

        #Look at X with names
        self.df_features = DataFrame(self.X.toarray(), columns=self.vectorizer.get_feature_names())


        #How many times each feature is non-zero
        self.d_vocab = []
        for column in self.df_features.columns:
            self.d_vocab.append([column, len(self.df_features[column].to_numpy().nonzero()[0])])


        #How many times each feature is non-zero - another way
        #d_vocab = []
        #for word in vectorizer.vocabulary_:
        #    d_vocab.append([word, len(np.where(X.toarray()[:, vectorizer.vocabulary_.get(word)] != 0)[0])])

        self.feature_nonzero = DataFrame(self.d_vocab,
                                         columns=["domain", "count"]
                                         ).sort_values(by="count", ascending=False)

        return self.X, self.y, self.names



