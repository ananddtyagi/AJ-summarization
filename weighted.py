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

def sentence_to_embeddings(article):

    embeddings = embed(article)

    return embeddings

# Read the file with weights
# Assume weights.txt has values of the vector seperated by commas
# Example. 3,1,4,1,3,5
def read_weight_vector():
    # with open("./logs/weights.txt") as file:
    #     vector = file.read().split(",")
    #
    #     for i in range(0, len(vector)):
    #         vector[i] = float(vector[i])
    #     # return vector
    return [.26520689655172416, .7347931034482759, 0.0, 0.0, 0.0, 0.0]

def factor_in_weights(weight_vector, sentence_score_list):

    #Add weight to first sentence
    sentence_score_list[0] += weight_vector[0]
    #
    # Remove first sentence from list
    # sentence_score_list.pop(0)

    #Add weight to other sentences
    for i in range(1,len(sentence_score_list)):
        index_per = int((i+1)/len(sentence_score_list))

        if index_per < 0.04: #1-4
            sentence_score_list[i] += weight_vector[1]
        elif index_per < 0.07: #4-7
            sentence_score_list[i] += weight_vector[2]
        elif index_per < 0.1: #7-10
            sentence_score_list[i] += weight_vector[3]

    # #Add weight to first sentence
    # sentence_score_list[0] += (weight_vector[0] * len(sentence_score_list))
    #
    # #Add weight to other sentences
    # for i in range(1,len(sentence_score_list)):
    #     index_per = i/len(sentence_score_list)
    #
    #     if(index_per >= 0 and index_per < 0.1):
    #         # 0-10
    #         sentence_score_list[i] += (weight_vector[1] * len(sentence_score_list))
    #     elif(index_per >= 0.1 and index_per < 0.2):
    #         # 10-20
    #         sentence_score_list[i] += (weight_vector[2] * len(sentence_score_list))
    #     elif(index_per >= 0.2 and index_per < 0.8):
    #         # 20-80
    #         sentence_score_list[i] += (weight_vector[3] * len(sentence_score_list))
    #     elif(index_per >= 0.8 and index_per < 0.9):
    #         # 80-90
    #         sentence_score_list[i] += (weight_vector[4] * len(sentence_score_list))
    #     else:
    #         # 90-100
    #         sentence_score_list[i] += (weight_vector[5] * len(sentence_score_list))

    return sentence_score_list

def similarity_score(embeddings, weight_vector):

    sparse_mat = sparse.csr_matrix(embeddings)
    similarities = cosine_similarity(sparse_mat)

    scores = numpy.sum(similarities, axis=1)
    #scores[0] += 20

    #weight[i] * len(scores)

    return scores

def first(scores, sentences):
    tup_scores = []
    for i, score in enumerate(scores):
        tup_scores.append((score, i))

    ordered_scores = sorted(tup_scores, reverse = True, key=lambda s: s[0]) #sorted in descending order by the score
    return sentences[ordered_scores[0][1]]

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

    for i, line in enumerate(file):
        if i >= 87000:
            json_article = json.loads(line)
            reference_summary = json_article["summary"] #extract all sentences from article
            reference_list.append(reference_summary)

    final_list = []
    for i in trange(len(summary_list), desc='Results File'):
        obj = {
            "reference" : reference_list[i],
            "system": summary_list[i]
        }

        final_list.append(obj)

    print(len(final_list))

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

    print('summarize')
    if os.path.exists('./logs/summary_list.txt'):
        print('previously completed')
        with open('./logs/summary_list.txt', 'rb') as file:
             summary_list = pickle.load(file)
    else:
        #Fetching weight vector
        weight_vector = read_weight_vector()

        print(weight_vector)
        t = tqdm(cleaned_articles, desc = 'Article 0:')
        for i, article in enumerate(t):
            t.set_description('Article %i' % i)

            if i >= 87000:
                embeddings = sentence_to_embeddings(article) #THIS SHOULD BE A CLEANED ARTICLE
                weights = numpy.multiply(weight_vector, len(article))

                sim_scores = similarity_score(embeddings)

                sim_scores = factor_in_weights(weights, sim_scores)
            # sim_scores = similarity_score(embeddings, weight_vector)

                summary_list.append(first(sim_scores, extracted_articles[i]))

        debug_logger('summary_list', summary_list)

    print(len(summary_list))
    write_results_file(summary_list)

main()



