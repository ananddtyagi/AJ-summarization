#code written by Anand Tyagi

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #removes tf debugging

import tensorflow as tf
import tensorflow_hub as hub

import jsonlines
from nltk import sent_tokenize, word_tokenize

def extract():
    with open('article_ex.txt') as file:
        sentences = sent_tokenize(file.read()) #extract all sentneces from article

    return sentences

def clean(sentences):
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

def sentence_to_embeddings(sentences):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    embeddings = embed(sentences)

    return embeddings

def similarity_score(embeddings):
    from sklearn.metrics.pairwise import cosine_similarity

    similarity = []
    for embedding_1 in embeddings:
        similarity.append(0)
        for embedding_2 in embeddings:
            #MAKE THIS INTO A TRIANGLE
            similarity[-1] += cosine_similarity([embedding_1, embedding_2])[0][1]

    return similarity

def order(scores, sentences):
    tup_scores = []
    for i, score in enumerate(scores):
        tup_scores.append((score, i))

    ordered_scores = sorted(tup_scores, reverse = True, key=lambda s: s[0]) #sorted in descending order by the score
    ordered = []
    for score in ordered_scores:
        ordered.append(sentences[score[1]])

    return ordered

def summarization(sentences, summary_length):

    return print(''.join(sentences[:summary_length]))

def debug_logger(process, x):
    print(process)
    file = open('output.txt', 'w')
    file.seek(0)
    file.write(str(x))
    file.close()
    print('debug logged')
    return

def main():
    sentences = []

    sentences = extract()
    debug_logger('sentences', sentences)
    cleaned_sentences = clean(sentences)
    debug_logger('cleaned_sentences', cleaned_sentences)
    embeddings = sentence_to_embeddings(cleaned_sentences)
    debug_logger('embeddings', embeddings)
    similarity = similarity_score(embeddings)
    debug_logger('similarity', similarity)
    ordered_sentences = order(similarity, sentences)
    debug_logger('ordered_sentences', ordered_sentences)

    summary_length = 1

    summary = summarization(ordered_sentences, summary_length)

    print(summary)
    return

main()



