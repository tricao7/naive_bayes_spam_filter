'''
Tri Cao
October 22, 2021
'''

import time
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import SnowballStemmer
from sklearn.metrics import confusion_matrix
import math, string, random, os
from collections import defaultdict

class LabeledData:

    def __init__(self, ham_path = 'data/2002/easy_ham', spam_path = 'data/2002/spam', X = None, y = None):
        '''
        This function will initialize the LabeledData object and create vectors for the data if 
        it is not passed into the function.
        Parameters:
        :ham_path: a str, this is the path of the ham emails that will be used with default path 'data/2002/easy_ham'
        :spam_path: a str, this is the path of the spam emails that will be used with default path 'data/2002/spam'
        :X: a list, the vectors of emails that will be used with default value of None
        :y: a list, the vectors of values determining whether or not the eamil is spam. 0 is ham, 1 is spam. Default value of None
        Returns:
        None
        '''
        self.ham_path = ham_path; self.spam_path = spam_path
        self.X = X; self.y = y
        if self.X == None:
            self.X = []; self.y = []
            files = os.listdir(self.ham_path)
            for file in files:
                x = LabeledData.parse_message('self',self.ham_path + '/' + file)
                self.X.append(x)
                self.y.append(0)
            files = os.listdir(self.spam_path)
            for file in files:
                x = LabeledData.parse_message('self',self.spam_path + '/' + file)
                self.X.append(x)
                self.y.append(1)

    @staticmethod
    def parse_line(line):
        '''
        This static method function will go through the line and remove the header line (a line with a single word
        followed by a colon, i.e. Date:). It will then strip the non-header line and return it.
        Parameters:
        :line: a str, this is the string of the line that will be read.
        Returns:
        :'': an empty str, if the line is a header-line it will be converted to an empty string.
        :line.strip(): a str, this is the line after it has been stripped given that it is not a header line.
        
        '''
        if ':' in line:
            split_line = line.strip().split(':')[0].split()
            if len(split_line) != 1:
                return line.strip()
            else:
                return ''
        else:
            return line.strip()
    
    def parse_message(self, fname):
        '''
        This function will open the file and read the lines. It will then select the subject line and body from the email and 
        and return them after it has been formatted.
        Parameters:
        :fname: a str, this is the name of the file that will be opened and read.
        Returns:
        :results: a str, this is the string of the contents of the file after it has been formatted.
        '''
        results = ''
        file = open(fname, errors = 'ignore', encoding = 'ascii')
        line = file.readline()
        for line in file:
            if 'Subject:' in line:
                subject = ['Subject:' ,'Re:', 're:']
                line = line.replace(subject[0],'').replace(subject[1],'').replace(subject[2],'')
                line = LabeledData.parse_line(line)
                results += line
            elif line =='\n':
                line = ''
                break
            else:
                line = ''
        for line in file:
            if line != '\n':
                line = LabeledData.parse_line(line)
                if len(line) != 0:
                    results +=  ' ' + line
        return results.strip()

class NaiveBayesClassifier:
    '''
    A class containing a trained naive Bayes classifier.
    '''
    def __init__(self, labeled_data, pseudocount = 0.5, max_words = 50):
        '''
        This function will take the data and create a dictionary containing the probabilites of the token correesponding
        to a spam or ham email.
        Parameters:
        :labeled_data: a list of lists, this is the list that contains the data for the email contents and whether or not
        the email is spam or ham.
        :psudocount: a float, this is the float of the pseudocount
        :max_words: an int, this is the number of words that will be used to select random words from get_tokens
        '''
        self.labeled_data = labeled_data
        self.total_spam = self.labeled_data.y.count(1)
        self.total_ham = self.labeled_data.y.count(0)
        self.max_words = max_words
        self.stemmer = SnowballStemmer('english')
        self.word_probs = self.count_words()
        for token in self.word_probs.keys():
            self.word_probs[token][0] = (self.word_probs[token][0]+pseudocount)/(self.total_spam + 2*pseudocount)
            self.word_probs[token][1] = (self.word_probs[token][1]+pseudocount)/(self.total_ham + 2*pseudocount)

    def count_words(self):
        '''
        This function will create a dictionary with the tokens as keys and incriment the values based on whether or not it is
        a spam or ham. Each key has a list of length 2. The first being the spam and the second ham.
        Parameters:
        None
        Returns:
        :x: a dict, this is the dictionary that has the token counts for the spam and ham emails.
        '''
        x = defaultdict(lambda: [0,0])
        data = self.labeled_data.X
        for i in range(len(data)):
            to_tokens = self.tokenize(data[i])
            for token in to_tokens:
                if self.labeled_data.y[i] == 1:
                    x[token][0] += 1
                else:
                    x[token][1] += 1 
        x = dict(x)
        return x

    def tokenize(self, email):
        '''
        This function will take a string and create tokens out of it. It will then remove the stop words that have
        little meaning in sentences.
        Parameters:
        :email: a str, this is a string containing an email that we will tokenize
        Returns:
        :set(tokens): a set, this is the set of tokens that was taken from the input email.
        '''
        email = email.lower().replace("n't", '')
        punct = string.punctuation; digits = string.digits
        remove_vals = punct + digits
        email = [ch for ch in email if ch not in remove_vals]
        email_words = ''
        email_words = list(set(email_words.join(email).split()))
        new_set = []
        for word in email_words:
            new_set.append(self.stemmer.stem(word))
        new_set = set(new_set)
        tokens = [word for word in new_set if word not in ENGLISH_STOP_WORDS]
        return set(tokens)

    def get_tokens(self, tv):
        '''
        This function will take a token vector and select random tokens from the vector. 
        Parameters:
        :tv: a set, this is the set of tokens that will be used to randomly select tokens
        Returns:
        :words: a list, this is the list of tokens from tv that was randomly selected.
        '''
        if self.max_words > len(tv):
            words = random.sample(sorted(tv), len(tv))
        else:
            words = random.sample(sorted(tv), self.max_words)
        return words

    def spam_probability(self, email):
        '''
        This function will find the probabilty of an email being spam using bayes' theorem. It will tokenize the email,
        get a random sample of tokens, and then return the probability that the email is spam.
        Parameters:
        :email: a str, this is the email that will be used to determine if it is spam or not.
        Returns:
        :prob: a float, this is the probability of the email being spam.
        '''
        to_tokens = self.tokenize(email)
        log_spam = 0; log_ham = 0
        ll_spam = 0; ll_ham = 0
        bool1 = True
        for token in to_tokens:
            if token in self.word_probs.keys():
                bool1 = False
                break
        if bool1 == True:
            return 1
        tokens = self.get_tokens(to_tokens)
        for token in tokens:
            if token in self.word_probs.keys():
                log_spam += math.log(self.word_probs[token][0])
                log_ham += math.log(self.word_probs[token][1])
        ll_spam = math.exp(log_spam);ll_ham = math.exp(log_ham)
        prob = (ll_spam*0.5)/(ll_spam*0.5 + ll_ham*0.5)
        return prob
    
    def classify(self, email):
        '''
        This function will determine if an email is likely to be spam or not and classify it as being more likely 
        spam or ham.
        Parameters:
        :email: a str, this is the email that will be input to classify it
        Returns:
        :True: a bool, this will be returned if the email is more likely to be spam
        :False: a bool, this will be returned if the email is more likely to be ham
        '''
        prob = self.spam_probability(email)
        if prob >= 0.5:
            return True
        else:
            return False

    def predict(self, X):
        '''
        This function will take a list object and go through it to see if each email is spam or not. It will
        then append its results to a list.
        Parameters:
        :X: a list, this is the list that contains all of the emails in the Naive Bayes Classifier
        Returns:
        :lst: a list, this is the list containing the boolean values of whether or not an email is spam.
        '''
        lst = []
        for email in X:
            lst.append(self.classify(email))
        return lst

def main():
    '''
    This function will create a training and testing set of the data and train the classifier.
    It will then create predictions for the testing data and create a confusion matrix based on the predictions
    and testing data. It determines the accurace of the confusion matrix and returns a string containing the 
    confusion matrix and the accuracy.
    Parameters:
    None
    Returns:
    :print(string): a str, this is the confusion matrix and the accuracy of the model.
    '''
    random.seed(25)
    training = LabeledData()
    testing = LabeledData(ham_path= 'data/2003/easy_ham', spam_path= 'data/2003/spam')
    classifier = NaiveBayesClassifier(training, max_words= 25)
    predictions = classifier.predict(testing.X)
    cf = confusion_matrix(testing.y, predictions)
    true_vals = 0
    totals = 0
    for i in range(len(cf)):
        for j in range(len(cf[i])):
            totals += cf[i][j]
            if i == j:
                true_vals += cf[i][j]
    accuracy = round((true_vals/totals)*100,2)
    string = str(cf) +'\naccuracy: '+str(accuracy)+'%'
    return print(string)

if __name__ == "__main__":
    main()
