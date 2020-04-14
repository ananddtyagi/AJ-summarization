
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
        sentences = sent_tokenize(file.read()) #extract all sentneces from article

    return sentences

def clean_sentences(sentences):
    import re
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    cleaned_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tokens = [w.lower() for w in tokens] #lowercase all tokens in each sentence
        tokens = [w for w in tokens if not w in stop_words] #remove stop words

        sentence = " ".join(tokens)
        sentence = re.sub(r'[^\w]', ' ', sentence) #remove all punctuation
        sentence = sentence.replace('   ', ' ') #the punctuation step adds spaces, to remove that without removing all spaces, I (Anand) added this step
        cleaned_sentences.append(sentence)

    return cleaned_sentences

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
    sentences = []
    sentences = extract_sentences()
    cleaned_sentences = clean_sentences(sentences)

    return

main()



