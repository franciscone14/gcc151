import os
import pandas as pd
import numpy as np
import spacy

from franc_lib.lexical.preprocessing import Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

class Semantics:

    def __init__(self, path="trainset/", trained_path="", trainset=0.8):
        """
           Construtor da semantica

           Keywords arguments:

           * path -- the path for the training set folder (default="trainset/")

           * trainset --  the percent of data that will be used for training (default=0.8)

           * trained_path -- the path for a file with the trained algorithm
        """
        self.path = path
        self.trainset = trainset
        self.trained_path = trained_path
        self.spacy_nlp = spacy.load('pt_core_news_sm')
        self.stopwords = spacy.lang.pt.stop_words.STOP_WORDS
        self.normalizer = Preprocessing()
        self.transformer = TfidfVectorizer()

    def read_data(self) -> dict:
        """
            read_data -- Reads the corpus path passed in the constructor and save the data

            Returns:

            * dataset -- dict {polarity, bin_polarity, review, set}, where (polarity: 0-5), (bin_polarity:0-1), (review: str), set(train, test)
        """
        dataset = {'polarity':[], 'bin_polarity': [], 'review':[], 'set':[]}

        for product in os.listdir(self.path):
            for score in os.listdir(self.path + product):
                for file in os.listdir(self.path + product + "/" + score + "/"):
                    if file.endswith('.txt'):
                        with open(self.path + product + "/" + score + "/" + file) as text_file:
                            comment = ""
                            for line in text_file.readlines():
                                if line != '\n':
                                    comment += line + " "
                            dataset['polarity'].append(float(score))
                            dataset['bin_polarity'].append(0 if float(score) < 3.0 else 1)
                            dataset['review'].append(comment.replace('\n',''))
                            dataset['set'].append('train' if np.random.rand() < self.trainset else 'test')
        
        return dataset

    def create_pandas_df(self, dataset) -> pd.DataFrame:
        """
            Receives the dataset and returns a pandas DataFrame
        """
        return pd.DataFrame(data=dataset)

    def preprocessing(self, text):
        text = self.normalizer.lowercase(text)
        text = self.normalizer.remove_punctuation(text)
        tokens = self.normalizer.tokenize_words(text)
        tokens = [token for token in tokens if token not in self.stopwords]
        return ' '.join(tokens)
    
    def apply_preprocessing(self, dataframe):
        dataframe['normalized_review'] = dataframe['review'].apply(self.preprocessing)
        return dataframe
    
    def feature_extraction(self, dataframe):
        train_reviews = dataframe[dataframe['set'] == 'train']['normalized_review'].values.tolist()
        train_classes = dataframe[dataframe['set'] == 'train']['polarity'].values.tolist()
        test_reviews = dataframe[dataframe['set'] == 'test']['normalized_review'].values.tolist()
        test_classes = dataframe[dataframe['set'] == 'test']['polarity'].values.tolist()

        self.transformer.fit(train_reviews)
        X = self.transformer.transform(train_reviews)
        X_test = self.transformer.transform(test_reviews)

        return (X, X_test, train_classes, test_classes)
    
    def train(self):
        dataset = self.read_data()
        self.dataframe = self.create_pandas_df(dataset)
        self.dataframe = self.apply_preprocessing(self.dataframe)

        self.X, self.X_test, self.train_classes, self.test_classes = self.feature_extraction(self.dataframe)

        self.classifier = LogisticRegression(n_jobs=3, verbose=True)
        print("Trainning....")
        self.classifier.fit(self.X, self.train_classes)
        print("Finished !")
        print("Precisão %.4f" % (accuracy_score(self.test_classes, self.classifier.predict(self.X_test)) * 100))
        print(confusion_matrix(self.test_classes, self.classifier.predict(self.X_test)))

    def get_confusion_matrix(self):
        m = confusion_matrix(self.test_classes, self.classifier.predict(self.X_test))
        print('v/v\t 0.0\t 1.0\t 2.0\t 3.0\t 4.0\t 5.0')
        print('0.0\t %d\t %d\t %d\t %d\t %d\t %d\t' % (m[0][0], m[0][1], m[0][2], m[0][3], m[0][4], m[0][5]))
        print('1.0\t %d\t %d\t %d\t %d\t %d\t %d\t' % (m[1][0], m[1][1], m[1][2], m[1][3], m[1][4], m[1][5]))
        print('2.0\t %d\t %d\t %d\t %d\t %d\t %d\t' % (m[2][0], m[2][1], m[2][2], m[2][3], m[2][4], m[2][5]))
        print('3.0\t %d\t %d\t %d\t %d\t %d\t %d\t' % (m[3][0], m[3][1], m[3][2], m[3][3], m[3][4], m[3][5]))
        print('4.0\t %d\t %d\t %d\t %d\t %d\t %d\t' % (m[4][0], m[4][1], m[4][2], m[4][3], m[4][4], m[4][5]))
        print('5.0\t %d\t %d\t %d\t %d\t %d\t %d\t' % (m[5][0], m[5][1], m[5][2], m[5][3], m[5][4], m[5][5]))

    def save_model(self):
        joblib.dump(self.classifier, 'saved_model.pkl') 

    def load_model(self):
        self.classifier = joblib.load('saved_model.pkl')

    def sentiment_analysis(self, text):
        # if(not os.path.isfile('saved_model.pkl')):
        self.train()
        self.save_model()
        # else:
        #     self.load_model()
        
        X = self.transformer.transform([self.preprocessing(text)])
        return self.classifier.predict(X)
