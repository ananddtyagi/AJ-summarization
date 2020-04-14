
#removes tf debugging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_hub as hub

import jsonlines

def extract_sentences():
    with jsonlines.open('input.jsonl') as reader:
        for obj in reader:
            print(obj)
            break
    return

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
    extract_sentences()
    return

main()



