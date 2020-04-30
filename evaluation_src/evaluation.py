import rouge_l
import json

def read_input():
    file = open("../data.txt", "r")

    list_articles = []
    for line in file:
        json_article = json.loads(line)

        list_articles.append(json_article)

    return list_articles

def aggregate_scores(list_articles):

    recall_total = 0
    precision_total = 0
    f_measure_total = 0

    for obj in list_articles:
        reference_sum = obj[0]["reference"]
        system_sum = obj[0]["system"]

        #Reference first, then system
        result = rouge_l.main(reference_sum, system_sum)

        recall_total += result["recall"]
        precision_total += result["precision"]
        f_measure_total += result["f-measure"]

    print("Recall: ", recall_total/len(list_articles))
    print("Precision: ", precision_total/len(list_articles))
    print("F-measure: ", f_measure_total/len(list_articles))

def main():
    list_articles = read_input()

    #print(list_articles)
    aggregate_scores(list_articles)

main()