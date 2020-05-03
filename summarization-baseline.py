# A summarization baseline. Takes the first sentence of each article and have that be the summary

import json
import nltk

#import jsonlines
from nltk import sent_tokenize, word_tokenize

def extract():
    file = open('./input_data/dev.jsonl', "r")

    articles = []
    i = 0
    for line in file:
        json_article = json.loads(line)

        sentences = sent_tokenize(json_article["text"]) #extract all sentences from article
        

        reference_sum = json_article["summary"]
        system_sum = sentences[0]
        
        obj = {
            "reference": reference_sum,
            "system": system_sum
        }

        articles.append(obj)

        if i % 100:
            print("Finish article ", i)
        
        i+=1

    return articles

def write_results_file(articles):
    

    with open('data-baseline.txt', 'w') as outfile:
        json.dump(articles, outfile)

    return

def main():
    articles = extract()

    write_results_file(articles)

main()