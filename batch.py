#code written by Anand Tyagi

import os
import json
import nltk
import ast #for reading from debug file
import sys
import pickle
import numpy
from tqdm import tqdm
from tqdm.auto import trange
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
# nltk.download('stopwords')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #removes tf debugging

import tensorflow as tf
import tensorflow_hub as hub

#import jsonlines
from nltk import sent_tokenize, word_tokenize
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


def similarity_score(embeddings):

    articles_similarity = []
    scores = []
    sparse_mat = []
    similarities = []
    for e in tqdm(embeddings, desc='Sim Score Calc'):
        sparse_mat = sparse.csr_matrix(e)
        similarities = cosine_similarity(sparse_mat)
        scores = numpy.sum(similarities, axis=1)
        scores[0] += 10

        articles_similarity.append(scores)

    return articles_similarity


def first(batch_scores, batch_sentences):

    summaries = []
    for i, scores in tqdm(enumerate(batch_scores)):

        tup_scores = []
        for j, score in enumerate(scores):
            tup_scores.append((score, j))

        ordered_scores = sorted(tup_scores, reverse = True, key=lambda s: s[0]) #sorted in descending order by the score

        summaries.append(batch_sentences[i][ordered_scores[0][1]])
    return summaries

def debug_logger(process, x):
    print(process)
    with open('./logs/' + process + '.txt', 'wb') as file:
        pickle.dump(x, file)
    print('debug logged')
    return

def write_results_file(summary_list): #added by Justin Chen
    file = open('./input_data/dev.jsonl', "r")

    #Take the answer list
    reference_list = []

    for line in file:
        json_article = json.loads(line)
        reference_summary = json_article["summary"] #extract all sentences from article
        reference_list.append(reference_summary)
    i = 0

    final_list = []
    for i in trange(len(summary_list), desc='Results File'):

        obj = {
            "reference" : reference_list[i],
            "system": summary_list[i]
        }

        final_list.append(obj)

    with open('./output_data/data.txt', 'w') as outfile:
        json.dump(final_list, outfile)

    print('finished writing results')
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
    # summary_file = open("./logs/summary.txt", 'w')
    # with open("./logs/summary.txt", 'r') as file:
    #     for i, line in enumerate(file):
    #         summary_list.append(line)
    #         if i == 108000: #wherever it broke
    #             break;
    # with open("./logs/summary.txt", 'r') as file:
    #     for line in tqdm(file):
    #         summary_list.append(line)


    print('summarize')
    if os.path.exists('./logs/summary_list.txt'):
        print('previously completed')
        with open('./logs/summary_list.txt', 'rb') as file:
             summary_list = pickle.load(file)
    else:
        t = tqdm(extracted_articles, desc = 'Article 0:')

        batch_size = 10000
        batch = []
        for i, article in enumerate(t):

            # if i > 108000: #if it broke
            t.set_description('Article %i' % i)
            batch.append(article)

            if i % batch_size == 0:
                embeddings = sentence_to_embeddings(batch)
                articles_similarity = similarity_score(embeddings)
                summary_list += first(articles_similarity, batch)
                print('Batch ', int(i / batch_size))
                batch = []
        if len(batch) > 0:
            embeddings = sentence_to_embeddings(batch)
            articles_similarity = similarity_score(embeddings)
            summary_list += first(articles_similarity, batch)

        debug_logger('summary_list', summary_list)

    print(len(summary_list))
    write_results_file(summary_list)

main()



