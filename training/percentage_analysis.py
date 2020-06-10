#code written by Anand Tyagi

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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #removes tf debugging

import tensorflow as tf
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def extract_articles():
    file = open('./input_data/dev.jsonl', "r")

    articles = []

    for i, line in enumerate(tqdm(file, total=108836, desc="Article Extraction Progress")):
        json_article = json.loads(line)

        articles.append(sent_tokenize(json_article['text']))

    return articles

def extract_answers():
    file = open('./input_data/dev.jsonl', "r")

    answers = []

    for i, line in enumerate(tqdm(file, total=108836, desc="Answer Extraction Progress")):
        json_article = json.loads(line)

        answers.append(sent_tokenize(json_article['summary']))

    return answers


def clean(articles):
    import re
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    cleaned_articles = []

    file = open('clean.txt', 'w')
    for article in tqdm(articles, desc='Cleaning Progress'):
        cleaned_sentences = []
        for i, sentence in enumerate(article):
            tokens = word_tokenize(sentence)
            tokens = [w.lower() for w in tokens] #lowercase all tokens in each sentence
            tokens = [w for w in tokens if not w in stop_words] #remove stop words

            sentence = " ".join(tokens)
            sentence = re.sub(r'[^\w]', ' ', sentence) #remove all punctuation
            sentence = sentence.replace('   ', ' ') #the punctuation step adds spaces, to remove that without removing all spaces
            cleaned_sentences.append(sentence)
            # if i == 50: #only store first x sentences
            #     break;
        file.write(str(cleaned_sentences) + '\n')
        cleaned_articles.append(cleaned_sentences)
    file.close()
    print('clean saved')
    return cleaned_articles

def sentence_to_embeddings(articles):

    embeddings = embed(articles)

    return embeddings

def percentile_calc(embeddings):

    sparse_mat = sparse.csr_matrix(embeddings)
    similarities = cosine_similarity(sparse_mat)
    answer_sim = similarities[-1][:-1] #do not include the answer similarity to itself

    closest_index = numpy.argmax(answer_sim)

    return closest_index / len(answer_sim) #return the percentile the gold standard answer is in

    # if closest_index == 0: #first sentence
    #     return 0
    # elif percentile < 0.10: #0-1%
    #     return 1
    # elif percentile < 0.20: #1-2%
    #     return 2
    # elif percentile < 0.80: #3-4%
    #     return 3
    # elif percentile < 0.90: #4-5%
    #     return 4
    #
    # return 5 # > first 4%

def weight_calc(percentile):
    return

def debug_logger(process, x):
    print(process)
    with open('./logs/' + process + '.txt', 'wb') as file:
        pickle.dump(x, file)
    print('debug logged')
    return

def main():

    print('extract articles')
    if os.path.exists('./logs/extracted_articles.txt'):
        print('previously completed')
        with open('./logs/extracted_articles.txt', 'rb') as file:
            extracted_articles = pickle.load(file)
    else:
        extracted_articles = extract_articles()
        debug_logger('extracted_articles', extracted_articles)

    print('extract answers')
    if os.path.exists('./logs/extracted_answers.txt'):
        print('previously completed')
        with open('./logs/extracted_answers.txt', 'rb') as file:
            extracted_answers = pickle.load(file)
    else:
        extracted_answers = extract_answers()
        debug_logger('extracted_answers', extracted_answers)

    print('clean')
    if os.path.exists('./logs/cleaned_articles.txt'):
        print('previously completed')
        with open('./logs/cleaned_articles.txt', 'rb') as file:
            cleaned_articles = pickle.load(file)
    else:
        cleaned_articles = clean(extracted_articles)
        debug_logger('cleaned_articles', cleaned_articles)

    print('percentile')
    if os.path.exists('./logs/percentile.txt'):
        print('previously completed')
        with open('./logs/percentile.txt', 'rb') as file:
            percentile = pickle.load(file)
    else:
        percentile = []

        t = tqdm(cleaned_articles, desc = 'Article 0:')

        for i, article in enumerate(t):
            t.set_description('Article %i' % i)

            embeddings = sentence_to_embeddings(article + extracted_answers[i])
            #last entry in embeddings is the embeddings of the answer for that article
            percentile.append(percentile_calc(embeddings))

        debug_logger('percentile', percentile)


    # # weights = numpy.divide(weights, 87000)
    # NUM_PARAMS = 1
    # weights = [0]*NUM_PARAMS
    #
    # for percent in percentile:



main()
