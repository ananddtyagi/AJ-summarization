# A basic implementation of evuluation metric rouge-2 
# The basics of this metric uses bigrams as a way to evualate summary extraction
# Found more here: https://www.aclweb.org/anthology/W04-1013.pdf
# https://www.aclweb.org/anthology/D19-1051.pdf

# Author: Justin C (320834)

sys_str = "the cat was found under the bed"
ans_str = "the cat was under the bed"

def bi_gram(str):
    list_str = str.split(" ")
    bi_gram_dict = {}
    for i in range(0,len(list_str)-1):
        word_one = list_str[i]
        word_two = list_str[i+1]

        bi_gram_dict[word_one + " " + word_two] = True

    return bi_gram_dict

def get_correct(sys_dict, ans_dict):
    correct = 0
    keys = sys_dict.keys()

    for key in keys:
        if(ans_dict.get(key) != None):
            correct += 1

    return correct


def input(system_str, answer_str):
    sys_dict = bi_gram(system_str)
    ans_dict = bi_gram(answer_str)

    correct = get_correct(sys_dict, ans_dict)

    precision = correct/len(sys_dict)
    recall = correct/len(ans_dict)
    f_measure = 2/((1/precision) + (1/recall))

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-measure: ", f_measure)

input(sys_str, ans_str)
