#code written by Anand Tyagi

import os
import json
import nltk
# nltk.download('stopwords')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #removes tf debugging

import tensorflow as tf
import tensorflow_hub as hub

#import jsonlines
from nltk import sent_tokenize, word_tokenize

def extract():
    file = open('./input_data/dev.jsonl', "r")

    articles = []
    i = 0
    for line in file:
        json_article = json.loads(line)

        sentences = sent_tokenize(json_article["text"]) #extract all sentences from article
        articles.append(sentences)

        if i == 2:
            break
        
        i+=1

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
            sentence = sentence.replace('   ', ' ') #the punctuation step adds spaces, to remove that without removing all spaces, I (Anand) added this step
            cleaned_sentences.append(sentence)
        
        cleaned_articles.append(cleaned_sentences)

    return cleaned_articles

def sentence_to_embeddings(articles):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    embeddings = []

    i = 0

    print("Before embeding step")
    for sentences in articles:
        single_embedding = embed(sentences)

        embeddings.append(single_embedding)

        print('Embedding ', i)
        i+=1

    return embeddings

def similarity_score(embeddings):
    from sklearn.metrics.pairwise import cosine_similarity

    articles_similarity = []

    print("Before Similarity")
    i = 0
    for embed in embeddings:
        similarity = []
        for embedding_1 in embed:
            similarity.append(0)
            for embedding_2 in embed:
                #MAKE THIS INTO A TRIANGLE
                similarity[-1] += cosine_similarity([embedding_1, embedding_2])[0][1]

        print("Similarity ", i)
        i+=1

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
    file = open('output.txt', 'w')
    file.seek(0)
    file.write(str(x))
    file.close()
    print('debug logged')
    return

def write_results_file(summary_list):
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

    with open('data.txt', 'w') as outfile:
        json.dump(final_list, outfile)

    return

def main():
    articles = []

    articles = extract()
    # debug_logger('sentences', articles)
    cleaned_articles = clean(articles)
    #debug_logger('cleaned_sentences', cleaned_articles)
    embeddings = sentence_to_embeddings(cleaned_articles)
    #debug_logger('embeddings', embeddings)
    articles_similarity = similarity_score(embeddings)
    #debug_logger('similarity', similarity)
    ordered_sentences_list = order_embeds_in_list(articles_similarity, articles)
    #debug_logger('ordered_sentences', ordered_sentences)

    summary_length = 1

    summary_list = get_results(ordered_sentences_list, summary_length)

    write_results_file(summary_list)

    # print(summary_list[0])

    # print(len(summary_list))
    

main()



