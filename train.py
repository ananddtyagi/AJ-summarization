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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #removes tf debugging

import tensorflow as tf
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def extract_articles():
    file = open('./input_data/dev.jsonl', "r")

    articles = []

    for i, line in enumerate(tqdm(file, total=108836, desc="Extraction Progress")):
        json_article = json.loads(line)

        articles.append(sent_tokenize(json_article['text']))

    return articles

def extract_sentences(articles):

    tokenized_articles = []

    for article in tqdm(articles, desc='Tokenization Progress'):
        tokenized_articles.append(sent_tokenize(article))

    return tokenized_articles


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

    embeddings = []
    # for i, _ in enumerate(tqdm(articles, desc='Initializing embeddings')):
    #     embeddings[i] = numpy.zeros((len(_), 512))

    for i, sentences in enumerate(tqdm(articles, desc='Embedding Progress')):
        embeddings.append(embed(sentences))

    return embeddings


def weight_calc(embeddings):

    articles_similarity = []
    scores = []
    sparse_mat = []
    similarities = []
    for e in tqdm(embeddings, desc='Sim Score Calc'):
        sparse_mat = sparse.csr_matrix(e)
        similarities = cosine_similarity(sparse_mat)
        scores = numpy.sum(similarities, axis=1)
        scores[0] += 10
        #ADD WEIGHTS TO HERE SOMEHOW
        articles_similarity.append(scores)

    return articles_similarity


def main():

    print('extract articles')
    if os.path.exists('./logs/extracted_articles.txt'):
        print('previously completed')
        with open('./logs/extracted_articles.txt', 'rb') as file:
            extracted_articles = pickle.load(file)
    else:
        extracted_articles = extract_articles()
        debug_logger('extracted_articles', extracted_articles)

    # print('extract sentences')
    # if os.path.exists('./logs/extracted_sentences.txt'):
    #     print('previously completed')
    #     with open('./logs/extracted_sentences.txt', 'rb') as file:
    #         extracted_sentences = pickle.load(file)
    # else:
    #     extracted_sentences = extract_sentences(extracted_articles)
    #     debug_logger('extracted_sentences', extracted_sentences)

    print('clean')
    if os.path.exists('./logs/cleaned_articles.txt'):
        print('previously completed')
        with open('./logs/cleaned_articles.txt', 'rb') as file:
            cleaned_articles = pickle.load(file)
    else:
        cleaned_articles = clean(extracted_articles)
        debug_logger('cleaned_articles', cleaned_articles)

    summary_list = []


    weights = [0,0,0,0,0] #0-10, 10-20, 20-80, 80-90, 90-100


    

    print('summarize')
    if os.path.exists('./logs/summary_list.txt'):
        print('previously completed')
        with open('./logs/summary_list.txt', 'rb') as file:
             summary_list = pickle.load(file)
    else:
        t = tqdm(extracted_articles, desc = 'Article 0:')


        for i, article in enumerate(t):

            # if i > 108000: #if it broke
            t.set_description('Article %i' % i)

            embeddings = sentence_to_embeddings(article)

            sim_scores = similarity_score(embeddings)

            summary_list.append(first(sim_scores, article))
        debug_logger('summary_list', summary_list)

    print(len(summary_list))
    write_results_file(summary_list)

main()



