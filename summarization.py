
#removes tf debugging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_hub as hub

import jsonlines
from nltk import sent_tokenize, word_tokenize

def extract_sentences():
    # with jsonlines.open('dev.jsonl') as reader:
    #     i = 0
    #     for obj in reader:
    #         print("text ", obj['text'], "\n")
    #         print("summary", obj['summary'], "\n")
    #         if i == 2:
    #             break
    #         i = i+1
    with open('article_ex.txt') as file:
        sentences = sent_tokenize(file.read())
    for sentence in sentences:
        sentence = "".join(word_tokenize(sentence))
    return sentences

def clean_sentences():
    return

def setnence_to_embeddings(sentences):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    embeddings = embed(sentences)
    return embeddings

def similarity_matrix():
    return

def rank_sentences():
    return

def summarization():
    return

def main():
    sentences = extract_sentences()
    return

main()



