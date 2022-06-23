from Cranfield_collection_HW import stop_list
from nltk.text import TextCollection
from nltk.probability import FreqDist
import numpy as np
import math
import nltk, re, string, math
from nltk import word_tokenize
nltk.download('punkt')
## Processing stop words, punctuation, numbers in data


def remove_other(words):
    from Cranfield_collection_HW import stop_list
    stop_list = stop_list.closed_class_stop_words
    tokens = word_tokenize(words)
    new_word_list = []

    for token in tokens:
        if token in stop_list or token in string.punctuation or token == '.I' or token == '.W':
            continue
        elif not re.search(r'\d|\W', token):
            new_word_list.append(token)

    return new_word_list


## Read data file and generate data dictionary{index, word_list}
def read_file(file_path):
    with open(file_path, 'r')as fr:
        lines = fr.readlines()
    query_dict = {} ## Data storage dictionary
    # num = 0
    # sum = 1
    # size = len(lines)
    flag = 0
    query = ''
    index  = 1
    for i in range(len(lines)):
        line = lines[i].strip()
        # index = 1
        # query = ''

        if '.I' in line and i != 0:
            query_dict[index] = remove_other(query)
            index += 1
            query  = ''
        
        if '.W' in line:
            flag = 1

        if flag == 1 and '.I' not in line:
            query += line.strip()

    query_dict[index] = remove_other(query)
    return query_dict


## Calculate idf, input thesaurus, 
## output: idf value of each word in thesaurus（dict），
def compute_idf(query_dict):
    corpus = []
    for query in query_dict:
        corpus.append(' '.join(query_dict[query]))
        # print(' '.join(query_dict[query]))
    # print(corpus)
    # final_corpus = [nltk.word_tokenize(sent) for sent in corpus]
    query_corpus = TextCollection(corpus)
    query_idf = {}
    query_word_list = []

    for query in query_dict:
        query_word_list+=query_dict[query]
    # print(len(query_word_list))
    query_word_set = set(query_word_list)
    # print(len(query_word_set))

    for query in query_word_set:
        query_idf[query] = query_corpus.idf(query)
    # print(query_idf)
    return query_idf, query_corpus


## Calculate tf, multiply by idf, and find the tfidf input: 
## 1. The lexicon trained by the previous idf, 2. All queries
## Output: each query: tf of all words, dict set of dict
def compute_tf(query_corpus, query_dict, query_idf):
    query_tf_idf = {}
    for query in query_dict:
        word_list = query_dict[query]
        total_len = len(word_list)
        word_set = set(word_list)
        tmp_query_tf_idf = {}
        for word in word_set:
            tf_value = query_corpus.tf(word, ' '.join(word_list))
            tmp_query_tf_idf[word] = tf_value*query_idf[word]
        query_tf_idf[query] = tmp_query_tf_idf
    return query_tf_idf


## Calculate the cosine value of two vectors
def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom==0.0:
        return 0.5
    else:
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

## Double cycle
def main():
    query_dict = read_file('Cranfield_collection_HW/cran.qry')
    abstract_dict = read_file('Cranfield_collection_HW/cran.all.1400')

    query_idf, query_corpus = compute_idf(query_dict)
    abstract_idf, abstract_corpus = compute_idf(abstract_dict)

    query_tf_idf = compute_tf(query_corpus, query_dict, query_idf)
    abstract_tf_idf = compute_tf(abstract_corpus, abstract_dict, abstract_idf)


    output_result = []
    num = 0
    print("query tf idf size =  ", len(query_tf_idf))
    print("abstract tf idf size = ", len(abstract_tf_idf))
    for query in query_tf_idf:
        query_word_dict = query_tf_idf[query] 
        query_word_list = []
        for query_word in query_word_dict:
            query_word_list.append(query_word)
        cos_value  = {}
        # print(query_word_list)
        for abstract in abstract_tf_idf:
            abstract_word_dict = abstract_tf_idf[abstract] 
            abstract_word_list = []
            for abstract_word in abstract_word_dict:
                abstract_word_list.append(abstract_word)
            total_word_list = set(query_word_list+abstract_word_list)

            query_vector = []

            for word in total_word_list:
                if word in query_word_list:  
                    query_vector.append(query_word_dict[word]) 
                else:
                    query_vector.append(0.0)

            abstract_vector = []
            for word in total_word_list:
                if word in abstract_word_list:
                    abstract_vector.append(abstract_word_dict[word])
                else:
                    abstract_vector.append(0.0)
            sim = cos_sim(query_vector, abstract_vector)

            cos_value[abstract] = sim

        ## Sort several documents of a query				
        new_cos_value = sorted(cos_value.items(), key=lambda d: d[1], reverse=True)
        for cos in new_cos_value:
            output_result.append(str(query)+' '+str(cos[0])+' '+str(cos[1]))
        if num%100 == 0:
            print(num)
        num += 1 
    with open('output.txt', 'w') as fw:
        for out in output_result:
            fw.write(out+'\n')

if __name__ == '__main__':
    main()