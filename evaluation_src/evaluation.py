import rouge_l
import json
from rouge import Rouge
from tqdm import tqdm

rouge = Rouge()

import sys

def read_input():
    file = open("../output_data/data.txt", "r")

    list_articles = json.load(file)

    # list_articles = []
    # for line in file:
    #     json_article = json.loads(line)

    #     list_articles.append(json_article)

    return list_articles

def aggregate_scores(list_articles):

    rouge_1 = {
        "r": 0,
        "p": 0,
        "f": 0
    }

    rouge_2 = {
        "r": 0,
        "p": 0,
        "f": 0
    }

    rouge_l = {
        "r": 0,
        "p": 0,
        "f": 0
    }


    for i, obj in enumerate(tqdm(list_articles, total=len(list_articles))):

        reference_sum = obj["reference"]
        system_sum = obj["system"]


        try:
            result = rouge.get_scores(system_sum, reference_sum)[0]

            rouge_1["r"] += result["rouge-1"]["r"]
            rouge_1["p"] += result["rouge-1"]["p"]
            rouge_1["f"] += result["rouge-1"]["f"]

            rouge_2["r"] += result["rouge-2"]["r"]
            rouge_2["p"] += result["rouge-2"]["p"]
            rouge_2["f"] += result["rouge-2"]["f"]

            rouge_l["r"] += result["rouge-l"]["r"]
            rouge_l["p"] += result["rouge-l"]["p"]
            rouge_l["f"] += result["rouge-l"]["f"]
        except ValueError:
            h = 0
            #Do nothing

    len_article = len(list_articles)

    rouge_1["r"] = rouge_1["r"]/len_article
    rouge_1["p"] = rouge_1["p"]/len_article
    rouge_1["f"] = rouge_1["f"]/len_article

    rouge_2["r"] = rouge_2["r"]/len_article
    rouge_2["p"] = rouge_2["p"]/len_article
    rouge_2["f"] = rouge_2["f"]/len_article

    rouge_l["r"] = rouge_l["r"]/len_article
    rouge_l["p"] = rouge_l["p"]/len_article
    rouge_l["f"] = rouge_l["f"]/len_article

    print("Rouge-1")
    print(rouge_1)
    print("\n")
    print("Rouge-2")
    print(rouge_2)
    print("\n")
    print("Rouge-l")
    print(rouge_l)

def main():

    sys.setrecursionlimit(2500)
    list_articles = read_input()

    # print(list_articles[107217])

    aggregate_scores(list_articles)

main()