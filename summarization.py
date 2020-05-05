#code written by Anand Tyagi

import os
import json
import nltk
import ast #for reading from debug file
import sys
import pickle
# from progress.bar import IncrementalBar
from tqdm.auto import trange

# nltk.download('stopwords')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #removes tf debugging

import tensorflow as tf
import tensorflow_hub as hub

#import jsonlines
from nltk import sent_tokenize, word_tokenize

def extract_articles():
    file = open('./input_data/dev.jsonl', "r")

    articles = []

    for i, line in enumerate(file):
        json_article = json.loads(line)

        sentences = sent_tokenize(json_article["text"]) #extract all sentences from article
        articles.append(sentences)

    return articles

def extract_sentences(articles):

    articles = []

    for i in trange(len(articles), desc='Tokenization Progress'):
        articles.append(sent_tokenize(json_article["text"]))

    return articles


def clean(articles):
    import re
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    cleaned_articles = []


    for article in articles:
        cleaned_sentences = []
        for sentence in article:
            tokens = word_tokenize(sentence)
            tokens = [w.lower() for w in tokens] #lowercase all tokens in each sentence
            tokens = [w for w in tokens if not w in stop_words] #remove stop words

            sentence = " ".join(tokens)
            sentence = re.sub(r'[^\w]', ' ', sentence) #remove all punctuation
            sentence = sentence.replace('   ', ' ') #the punctuation step adds spaces, to remove that without removing all spaces
            cleaned_sentences.append(sentence)

        cleaned_articles.append(cleaned_sentences)

    return cleaned_articles

def sentence_to_embeddings(articles):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    embeddings = []
    single_embedding = []
    i = 0

    print("Before embeding step")
    for sentences in articles:
        embeddings.append(embed(sentences))
        print('Embedding ', i)
        i+=1

    return embeddings

def similarity_score(embeddings):
    from sklearn.metrics.pairwise import cosine_similarity

    articles_similarity = []

    print("Before Similarity")

    data_tqdm = trange(len(embeddings), desc="Data Progress")
    for e in data_tqdm:
        data_tqdm.set_description("Data Progress (article %i)", e)
        similarity = []
        embedding = embeddings[e]
        for i in trange(len(embeding), desc='Article Progress'):
            similarity = [0]*len(embeding)

            for j in range(i, len(embeding)):
                similarity[i] += cosine_similarity([embeding[i]], [embeding[j]])[0][0]
                similarity[j] += cosine_similarity([embeding[i]], [embeding[j]])[0][0]

        articles_similarity.append(similarity)

    return articles_similarity

def order(scores, sentences):
    tup_scores = []
    for i, score in enumerate(scores):
        tup_scores.append((score, i))

    ordered_scores = sorted(tup_scores, reverse = True, key=lambda s: s[0]) #sorted in descending order by the score
    ordered = []
    for score in ordered_scores:
        ordered.append(sentences[score[1]])

    return ordered

def order_embeds_in_list(score_list, article_list):

    ordered_list = []

    for i in range(0, len(score_list)):
        scores = score_list[i]
        sentences = article_list[i]

        ordered_sentences = order(scores, sentences)

        ordered_list.append(ordered_sentences)

    return ordered_list

def summarization(sentences, summary_length):

    return ''.join(sentences[:summary_length])

def get_results(ordered_sentences_list, length):
    results_summary_list = []

    for sentences in ordered_sentences_list:
        summary = summarization(sentences, length)

        results_summary_list.append(summary)

    return results_summary_list

def debug_logger(process, x):
    print(process)
    with open('./logs/' + process + '.txt', 'wb') as file:
        print(x)
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
    for i in range(0, len(summary_list)):

        obj = {
            "reference" : reference_list[i],
            "system": summary_list[i]
        }

        final_list.append(obj)

    with open('./output_data/data.txt', 'w') as outfile:
        json.dump(final_list, outfile)

    return



def main():
    articles = []

    print('extract articles')
    if os.path.exists('./logs/extracted_articles.txt'):
        print('previously completed')
        with open('./logs/extracted_articles.txt', 'rb') as file:
            extracted_articles = pickle.load(file)
    else:
        extracted_articles = extract_articles()
        debug_logger('extracted_articles', extracted_articles)

    print('extract sentences')
    if os.path.exists('./logs/extracted_sentences.txt'):
        print('previously completed')
        with open('./logs/extracted_sentences.txt', 'rb') as file:
            extracted_articles = pickle.load(file)
    else:
        extracted_sentences = extract_sentences()
        debug_logger('extracted_sentences', extracted_sentences)

    print('clean')
    if os.path.exists('./logs/cleaned_articles.txt'):
        print('previously completed')
        with open('./logs/cleaned_articles.txt', 'rb') as file:
            cleaned_articles = pickle.load(file)
    else:
        cleaned_articles = clean(articles)
        debug_logger('cleaned_articles', cleaned_articles)

    print('sen2emb')
    if os.path.exists('./logs/embeddings.txt'):
        print('previously completed')
        with open('./logs/embeddings.txt', 'rb') as file:
            embeddings = pickle.load(file)
    else:
        embeddings = sentence_to_embeddings(cleaned_articles)
        debug_logger('embeddings', embeddings)

    print('simscore')
    if os.path.exists('./logs/articles_similarity.txt'):
        print('previously completed')
        with open('./logs/articles_similarity.txt', 'rb') as file:
            articles_similarity = pickle.load(file)
    else:
        articles_similarity = similarity_score(embeddings)
        debug_logger('articles_similarity', articles_similarity)

    print('orederembilist')
    if os.path.exists('./logs/ordered_sentences_list.txt'):
        print('previously completed')
        with open('./logs/ordered_sentences_list.txt', 'rb') as file:
            ordered_sentences_list = pickle.load(file)
    else:
        ordered_sentences_list = order_embeds_in_list(articles_similarity, articles)
        debug_logger('ordered_sentences_list', ordered_sentences_list)

    # file = open('output.txt','r')
    # for line in file:
    #     ordered_sentences_list = ast.literal_eval(line)

    summary_length = 1
    print('getresults')
    summary_list = get_results(ordered_sentences_list, summary_length)

    write_results_file(summary_list)

    # print(summary_list[0])

    # print(len(summary_list))


main()



