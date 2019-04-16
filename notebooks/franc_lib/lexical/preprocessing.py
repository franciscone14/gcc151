import nltk
import unidecode
import string
import os

class Preprocessing:

    def __init__(self, save_path='../data/normilized/', file_name=None):
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        self.stemmer = nltk.stem.RSLPStemmer()

        self.save_path = save_path
        self.file_name = file_name

        if(not os.path.isdir(save_path)):
            os.makedirs(save_path)

    def remove_accents(self, text):
        return unidecode.unicode(text)
    
    def remove_punctuation(self, text):
        return text.translate(str.maketrans('','', string.punctuation))

    def tokenize_sentences(self, text):
        return self.sent_tokenizer.tokenize(text)

    def tokenize_words(self, text):
        return nltk.tokenize.word_tokenize(text)
    
    def lemmatize(self, text):
        return text

    def stemmize(self, tokens):
        return [self.stemmer.stem(word) for word in tokens]
    
    def normalization_pipeline(self, text, remove_accents=False, remove_punctuation=False, tokenize_sentences=False, tokenize_words=False, lemmatize=False, stemmize=False):
        text = self.remove_accents(text) if remove_accents else text
        file = self.file_name + '_no_accents.txt'
        self.save(file_name=file, data=text)
        text = self.remove_punctuation(text) if remove_punctuation else text
        file = self.file_name + '_no_accents.txt'
        self.save(file_name=file, data=text)
        text = self.tokenize_sentences(text) if tokenize_sentences else text
        file = self.file_name + '_no_accents.txt'
        self.save(file_name=file, data=text)
        text = self.tokenize_words(text) if tokenize_words else text
        file = self.file_name + '_no_accents.txt'
        self.save(file_name=file, data=text)
        text = self.lemmatize(text) if lemmatize else text
        file = self.file_name + '_no_accents.txt'
        self.save(file_name=file, data=text)
        text = self.stemmize(text) if stemmize else text
        
        return text
    
    def save(self, file_name=None, data=None):
        with open(os.path.join(self.path, file_name)) as file:
            for d in data:
                file.write('%s \n' % d)        