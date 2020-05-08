# A summarization baseline. Takes the first sentence of each article and have that be the summary

import json
import nltk
from tqdm import tqdm

#import jsonlines
from nltk import sent_tokenize, word_tokenize

def extract():
    file = open('./input_data/test.jsonl', "r")

    articles = []
    for i, line in enumerate(tqdm(file, total=108836)):
        json_article = json.loads(line)

        sentences = sent_tokenize(json_article["text"]) #extract all sentences from article


        reference_sum = json_article["summary"]
        system_sum = sentences[0]

        obj = {
            "reference": reference_sum,
            "system": system_sum
        }

        articles.append(obj)

    return articles

def write_results_file(articles):


    with open('./output_data/data-baseline.txt', 'w') as outfile:
        json.dump(articles, outfile)

    return

def main():
    articles = extract()

    write_results_file(articles)

main()