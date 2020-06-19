#code written by Anand Tyagi

#uses the rouge score between each sentence to evalute how relavent a setence is and uses the sum of that as the popularity score

import os
import json
from nltk import sent_tokenize, word_tokenize
import ast #for reading from debug file
import sys
import pickle
import numpy
from tqdm import tqdm
from tqdm.auto import trange
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from rouge import Rouge
rouge = Rouge()
input_data = '../input_data/train.jsonl'

#change to sys input
MAX_SEN = 100
START = 0

def extract_articles():
    file = open(input_data, "r")

    articles = []

    for i, line in enumerate(tqdm(file, total=MAX_SEN+START, desc="Article Extraction Progress")):
        if i >= START:
            if i == MAX_SEN+START:
                break;
            json_article = json.loads(line)

            articles.append(sent_tokenize(json_article['text']))
            if len(articles[-1]) == 1: #if it only finds one sentence:
                print(articles[-1])
                articles[-1] == articles[-1][0].split('\n\n') #I found this to be one of the common cases where the sentence tokenizer would fail
    return articles

def extract_answers():
    file = open(input_data, "r")

    answers = []

    for i, line in enumerate(tqdm(file, total=MAX_SEN+START, desc="Answer Extraction Progress")):
        if i >= START:
            if i == MAX_SEN+START:
                break;
            json_article = json.loads(line)

            answers.append(sent_tokenize(json_article['summary']))

    return answers


def clean(articles):
    import re
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    cleaned_articles = []

    for article in tqdm(articles, desc='Cleaning Progress'):
        cleaned_sentences = []
        for i, sentence in enumerate(article):
            tokens = word_tokenize(sentence)
            tokens = [w.lower() for w in tokens] #lowercase all tokens in each sentence
            tokens = [w for w in tokens if not w in stop_words] #remove stop words

            sentence = " ".join(tokens)
            print("1:", sentence)
            sentence = re.sub(r'[^\w]', ' ', sentence) #remove all punctuation
            print("2:",     sentence)
            sentence = sentence.replace('   ', ' ') #the punctuation step adds spaces, to remove that without removing all spaces
            cleaned_sentences.append(sentence)

        cleaned_articles.append(cleaned_sentences)
    return cleaned_articles

def weight_index_calc(sentences):
    #last index in sentences is the answer
    answer = sentences.pop(-1)
    max_score = 0
    closest_index = 0
    print(sentences)
    for i, sentence in enumerate(sentences):
        if rouge.get_scores(sentence, answer)[0]["rouge-l"]["f"] > max_score:
            closest_index = i

    if len(sentences) == 0: #one sentence article
        return 0

    percentile = (closest_index+1) / len(sentences)

    weights = [0,0,0,0,0,0] #0, 0-10, 10-20, 20-80, 80-90, 90-100

    if closest_index == 0: #first sentence
        return 0
    elif percentile < 0.10: #0-10 (excluding first sentence)
        return 1
    elif percentile < 0.20: #10-20
        return 2
    elif percentile < 0.80: #20-80
        return 3
    elif percentile < 0.90: #80-90
        return 4
    else: #90-100
        return 5

    return 0

def debug_logger(process, x):
    print(process)
    with open('../logs/train/' + process + '.txt', 'wb') as file:
        pickle.dump(x, file)
    print('debug logged')
    return

def main():

    if not os.path.isdir('../logs'):
        os.mkdir('../logs')
    if not os.path.isdir('../logs/train'):
        os.mkdir('../logs/train')

    print('extract articles')
    if os.path.exists('../logs/train/extracted_articles.txt'):
        print('previously completed')
        with open('../logs/train/extracted_articles.txt', 'rb') as file:
            extracted_articles = pickle.load(file)
    else:
        extracted_articles = extract_articles()
        # debug_logger('extracted_articles', extracted_articles)

    print('extract answers')
    if os.path.exists('../logs/train/extracted_answers.txt'):
        print('previously completed')
        with open('../logs/train/extracted_answers.txt', 'rb') as file:
            extracted_answers = pickle.load(file)
    else:
        extracted_answers = extract_answers()
        # debug_logger('extracted_answers', extracted_answers)

    assert len(extracted_articles) == len(extracted_answers)

    print('clean')
    if os.path.exists('../logs/train/cleaned_articles.txt'):
        print('previously completed')
        with open('../logs/train/cleaned_articles.txt', 'rb') as file:
            cleaned_articles = pickle.load(file)
    else:
        cleaned_articles = clean(extracted_articles)
        # debug_logger('cleaned_articles', cleaned_articles)

    weights = [0,0,0,0,0,0] #first sentence, 0-10 (not including the first sentence), 10-20, 20-80, 80-90, 90-100
    print(len(cleaned_articles))
    sys.setrecursionlimit(300 * 300 + 10)

    t = tqdm(cleaned_articles, desc = 'Article 0:')

    for i, article in enumerate(t):
        t.set_description('Article %i' % i)

        weights[weight_index_calc(cleaned_articles + extracted_answers[i])] += 1
        if i % 40000 == 0:
            print(i, " : ", list(weights))
    weights = numpy.divide(weights, len(cleaned_articles))
    print(list(weights))

    print(START)
    print(START + MAX_SEN)

main()
