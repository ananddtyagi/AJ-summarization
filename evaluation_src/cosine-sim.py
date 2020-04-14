# A basic implementation of evuluation metric through cosine similarity
# **NOTE** Not clear on how the output is going to be. For now, just use tf value instead of tf-idf

# Author: Justin C (320834)

import nltk
import math
# nltk.download('punkt')

#Test str, input strings for cosine similarity
sys_str = "the cat was found under the bed"
ans_str = "the cat was under the bed"

#System Answer Variables
ans_frequency_total = 0
ans_word_frequency_dict = {}
ans_word_document_dict = {}

def tokenize_ans(ans_str):
    token_words = nltk.word_tokenize(ans_str)

    ans_obj = {
        "ans_str": ans_str,
        "tokens": token_words,
        "tf": {}
    }

    return ans_obj
    
def get_ans_tf(ans_obj):
    for word in ans_obj["tokens"]:
        if(ans_obj["tf"].get(word) == None):
            ans_obj["tf"][word] = 1
        else:
            amount = ans_obj["tf"].get(word)
            ans_obj["tf"].update({word: amount+1})
    return ans_obj

def ans_func():
    ans_obj = tokenize_ans(ans_str)

    ans_obj = get_ans_tf(ans_obj)

    return ans_obj

#===============================================================================

def tokenize_sys(sys_str):
    token_words = nltk.word_tokenize(sys_str)

    sys_obj = {
        "ans_str": sys_str,
        "tokens": token_words,
        "tf": {}
    }

    return sys_obj
    
def get_sys_tf(sys_obj):
    for word in sys_obj["tokens"]:
        if(sys_obj["tf"].get(word) == None):
            sys_obj["tf"][word] = 1
        else:
            amount = sys_obj["tf"].get(word)
            sys_obj["tf"].update({word: amount+1})
    return sys_obj

def sys_func():
    sys_obj = tokenize_ans(sys_str)

    sys_obj = get_ans_tf(sys_obj)

    return sys_obj

#===============================================================================

def calculate_cosine(ans_obj, sys_obj):
    h = 0
    #Calculate numerator
    vector = []
    if len(ans_obj["tokens"]) >= len(sys_obj["tokens"]):
        vector = ans_obj["tokens"]
    else:
        vector = sys_obj["tokens"]
    
    numerator = 0
    for word in vector:
        if(ans_obj["tf"].get(word) != None and sys_obj["tf"].get(word) != None):
            numerator += ans_obj["tf"].get(word) * sys_obj["tf"].get(word)
        

    demoninator = 0

    ans_value = 0
    sys_value = 0

    for word in ans_obj["tokens"]:
        if(ans_obj["tf"].get(word) != None):
            ans_value += math.pow(ans_obj["tf"].get(word), 2)

    for word in sys_obj["tokens"]:
        if(sys_obj["tf"].get(word) != None):
            sys_value += math.pow(sys_obj["tf"].get(word), 2)

    demoninator = math.sqrt(ans_value * sys_value)

    if(demoninator == 0):
        return 0
    else:
        #print(numerator)
        #print(demoninator)
        return numerator/demoninator
    
def main():
    ans_obj = ans_func()
    sys_obj = sys_func()

    value = calculate_cosine(ans_obj, sys_obj)

    print(value)

main()