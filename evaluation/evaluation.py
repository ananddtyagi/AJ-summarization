import rouge_l
import json
from rouge import Rouge
from tqdm import tqdm

rouge = Rouge()

import sys
datafile = "../output_data/top2data.txt"

def read_input():
    file = open(datafile, "r")

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

    len_article = len(list_articles)
    skipped = 0
    total_used = 0
    t = tqdm(list_articles, desc = 'Eval:')

    for i, obj in enumerate(t):

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

            total_used += 1
        except ValueError:
            skipped += 1
            #Do nothing



    print("Final Scores")
    rouge_1["r"] = rouge_1["r"]/total_used
    rouge_1["p"] = rouge_1["p"]/total_used
    rouge_1["f"] = rouge_1["f"]/total_used

    rouge_2["r"] = rouge_2["r"]/total_used
    rouge_2["p"] = rouge_2["p"]/total_used
    rouge_2["f"] = rouge_2["f"]/total_used

    rouge_l["r"] = rouge_l["r"]/total_used
    rouge_l["p"] = rouge_l["p"]/total_used
    rouge_l["f"] = rouge_l["f"]/total_used

    print(datafile)
    print("f1")
    print("R1: ", rouge_1["f"] * 100, "R2: ", rouge_2["f"] * 100, "RL: ", rouge_l["f"] * 100)
    print("Skipped")
    print(skipped)
    print("Total Used")
    print(total_used)
    print("Rouge-1")
    print(rouge_1)
    print("\n")
    print("Rouge-2")
    print(rouge_2)
    print("\n")
    print("Rouge-l")
    print(rouge_l)

def main():

    list_articles = read_input()
    max_ref = 0
    max_sys = 0

    nonevalues = 0

    for _, i in enumerate(tqdm(list_articles)):
        if len(i['reference']) > max_ref:
            max_ref = len(i['reference'])
        if len(i['system']) > max_sys:
            max_sys = len(i['system'])
    sys.setrecursionlimit(max_ref * max_sys + 10)

    print(nonevalues)
    # print(list_articles[107217])

    aggregate_scores(list_articles)

main()