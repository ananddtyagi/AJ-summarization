# A basic implementation of evuluation metric rouge-l
# The basics of this metric LCS for determining the similarity between two summaries
#
# Note this is only for sentence level LCS
#
# Found more here: https://www.aclweb.org/anthology/W04-1013.pdf
# https://www.aclweb.org/anthology/D19-1051.pdf

# Author: Justin C (320834)

# Globals

import math

#Get the length of the longest common subsequence in X and Y
def lcs(X, Y, m, n): 

    # 2D Array to store memoized data
    L = [[0 for x in range(n+1)] for x in range(m+1)] 
  
    # Following steps build L[m+1][n+1] using bottom up approach
    for i in range(m+1): 
        for j in range(n+1): 
            if i == 0 or j == 0: 
                L[i][j] = 0
            elif X[i-1] == Y[j-1]: 
                L[i][j] = L[i-1][j-1] + 1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 
  
    return L[m][n]
    
def get_recall(ref_sen, lcs_len):
    return lcs_len / len(ref_sen)

def get_precision(sys_sen, lcs_len):
    return lcs_len / len(sys_sen)

def get_F_measure(recall, precision, B):

    num = (1+pow(B,2)) * recall * precision

    demon = recall + (pow(B,2) * precision)

    return num/demon

#Calling function, inputs reference sentence and system sentence, outputs the recall, precision and f-measure
def main(ref_sentence, sys_sentence):
    lcs_length = lcs(ref_sentence, sys_sentence, len(ref_sentence), len(sys_sentence))

    recall = get_recall(ref_sentence, lcs_length)
    precision = get_precision(sys_sentence, lcs_length)

    f_mes = get_F_measure(recall, precision, 1)

    #print(str(recall) + "\t" + str(precision) + "\t" + str(f_mes))

    return {
        "recall" : recall,
        "precision" : precision, 
        "f-measure" : f_mes
    }

