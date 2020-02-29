from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pickle import dump, load
import nltk
from re import match
from nltk.stem import SnowballStemmer
import numpy as np
import math

def get_text_string(fname):
    f = open(fname, encoding = "utf-8")
    text_string = f.read()
    f.close()

    soup = BeautifulSoup(text_string, 'lxml')
    text_string = soup.get_text()
    text_string = text_string.lower()

    return text_string


def get_raw_tokens(text_string):
    return nltk.word_tokenize(text_string)

def get_clean_tokens(raw_tokens):
    clean_tokens = []
    for tok in raw_tokens:
        t = []
        for ch in tok:
            if match(r'[a-záéíóúñA-ZÁÉÍÓÚÑ]', ch):
                t.append(ch)
        letterToken = ''.join(t)
        if letterToken != '':
            clean_tokens.append(letterToken)
    return clean_tokens;

def remove_stopwords(clean_tokens):
    norm_tokens = []
    for tok in clean_tokens:
        if(tok not in stopwords.words('spanish')):
            norm_tokens.append(tok)
    return norm_tokens

def get_norm_tokens(clean_tokens):
    norm_tokens = []
    clean_tokens = remove_stopwords(clean_tokens)
    stemmer = SnowballStemmer('spanish')
    for tok in clean_tokens:
        norm_tokens.append(stemmer.stem(tok))
    return norm_tokens

def save_file_object(fname, obj):
    output = open(fname, 'wb')
    dump(obj, output, -1)
    output.close()

def get_file_object(fname):
    input = open(fname, 'rb')
    file_object = load(input)
    input.close()

    return file_object

def get_vocabulary(norm_tokens):
    vocabulary = list(set(norm_tokens))
    vocabulary.sort()
    return vocabulary

def get_context(norm_tokens, palabra):
    context = []
    for x in range(0, len(norm_tokens)):
        if(norm_tokens[x] == palabra):
            for y in range(-4,4):
                if(y != 0 and (y + x >= 0) and (x + y < len(norm_tokens))):
                    context.append(norm_tokens[x + y])
    return context

def get_contexts(norm_tokens, vocabulary):
    contexts = dict()
    for term in vocabulary:
        contexts[term] = get_context(norm_tokens, term)
    return contexts

def get_vectors(contexts, vocabulary):
    vectors = dict()
    for term in vocabulary:
        vector = []
        for x in range(0, len(vocabulary)):
             vector.append(contexts[term].count(vocabulary[x]))
        vectors[term] = vector
    return vectors

def get_prom_vectors(vectors, vocabulary):
    prom_vectors = dict()
    for voc_main in vocabulary:
        vector = vectors[voc_main]
        div =  np.sum(np.array(vector))
        for i in range(0, len(vocabulary)):
            vector[i] = vector[i] / div
        prom_vectors[voc_main] = vector
    return prom_vectors

def get_frec_vector(contexts, vocabulary):
    frec_vector = []
    for voc_vec in vocabulary:
        cnt = 0
        for voc in vocabulary:
            if(voc_vec in contexts[voc]):
                cnt += 1
        frec_vector.append(cnt)
    return frec_vector

def get_v_tf(vocabulary, vector, k):
    v_tf = list()
    for i in range(0, len(vocabulary)):
        v_tf.append( ( (k + 1) * vector[i] ) / (vector[i] + k) )
    return v_tf

def get_avdl(vectors, vocabulary):
    avdl = 0    
    for w in vocabulary:
        avdl += np.sum(np.array(vectors[w]))
    avdl = avdl / len(vocabulary)
    return avdl

def get_bm25(vocabulary, vector, avdl):
    bm25 = list()
    b = 0.75
    nsum = np.sum(vector)
    for i in range(0, len(vocabulary)):
        bm25.append( ( (k + 1) * vector[i] ) / (vector[i] + k * ( 1 - b + (b * ( nsum / avdl))) ) )
    return bm25

def get_IDF(frec_vector, vocabulary):
    idf = list()
    for i in range(0, len(vocabulary)):
        idf.append(math.log((9057 + 1) / frec_vector[i]))
    return idf;
        
def get_ctx_vectors(vocabulary, vectors, k, frec_vector):
    ctx_vectors = dict()
    idf = get_IDF(frec_vector, vocabulary)
    for w in vocabulary:
        v_tf = get_v_tf(vocabulary, vectors[w], k)
        ctx_vectors[w] = np.multiply(np.array(v_tf),np.array(idf)).tolist()
    return ctx_vectors

def get_bm25_idf(vocabulary, vectors, k, frec_vector):
    bm25_idf = dict()
    idf = get_IDF(frec_vector, vocabulary)
    avdl = get_avdl(vectors, vocabulary)
    for w in vocabulary:
        bm25 = get_bm25(vocabulary, vectors[w], avdl)
        bm25_idf[w] = np.multiply(np.array(bm25),np.array(idf)).tolist()
    return bm25_idf

def get_cosines(word, vocabulary, fname_vectors):
    cosines = dict()
    vectors = get_file_object(fname_vectors)
    vec = vectors[word]
    vec = np.array(vec)
    for v in vocabulary:
        vector = vectors[v]
        vector = np.array(vector)
        cosine = np.dot(vec, vector) / ((np.sqrt(np.sum(vec **2))) * (np.sqrt(np.sum(vector ** 2))))
        cosines[v] = cosine
    import operator
    cosines = sorted(cosines.items(),
                     key = operator.itemgetter(1),
                     reverse = True)
    return cosines

def print_cosines(fname, cosines):
    f = open(fname, 'w')
    for item in cosines:
        for w in item:
            f.write(str(w))
            f.write(" ")
        f.write("\n")
    f.close()

#fname = "e961024.htm";
#text_str = get_text_string(fname)
#r_tokens = get_raw_tokens(text_str)
#clean_tokens = get_clean_tokens(r_tokens)

fname_tokens = "norm_tokens.txt"
#norm_tokens = get_norm_tokens(clean_tokens)
#save_file_object(fname_tokens, norm_tokens)
norm_tokens = get_file_object(fname_tokens)

fname_vocabulary = "vocabulary.txt"
#vocabulary = get_vocabulary(norm_tokens)
#save_file_object(fname_vocabulary, vocabulary)
vocabulary = get_file_object(fname_vocabulary)

fname_contexts = "contexts.txt"
#contexts = get_contexts(norm_tokens, vocabulary)
#save_file_object(fname_contexts, contexts)
contexts = get_file_object(fname_contexts)

fname_vectors = "vectors.txt"
#vectors = get_vectors(contexts, vocabulary)
#save_file_object(fname_vectors, vectors)
vectors = get_file_object(fname_vectors)

'''
fname_cosines = "def_cosines.txt"
def_cosines = get_cosines('mexic', vocabulary, fname_vectors)
print_cosines(fname_cosines, def_cosines)
'''

#print(vectors['mexic'])
print("\nnwn\n")

#fname_prom_vectors = "prom_vectors.txt"
#prom_vectors = get_prom_vectors(vectors, vocabulary)
#save_file_object(fname_prom_vectors, prom_vectors)
#prom_vectors = get_file_object(fname_prom_vectors)

'''
fname_cosines = "prom_cosines.txt"
prom_cosines = get_cosines('mexic', vocabulary, fname_prom_vectors)
print_cosines(fname_cosines, prom_cosines)
'''

#print(prom_vectors['mexic'])
print("\nnwn\n")

fname_frec_vector = "frec_vector.txt"
#frec_vector = get_frec_vector(contexts, vocabulary)
#save_file_object(fname_frec_vector, frec_vector)
frec_vector = get_file_object(fname_frec_vector)

fname_ctx_vectors = "ctx_vectors.txt"
k = 0.8
#ctx_vectors = get_ctx_vectors(vocabulary, vectors, k, frec_vector)
#save_file_object(fname_ctx_vectors, ctx_vectors)
#ctx_vectors = get_file_object(fname_ctx_vectors)

'''
fname_cosines = "ctx_cosines.txt"
ctx_cosines = get_cosines('mexic', vocabulary, fname_ctx_vectors)
print_cosines(fname_cosines, ctx_cosines)
'''

fname_BM25 = "bm25idf.txt"
#bm25idf = get_bm25_idf(vocabulary, vectors, k, frec_vector)
#save_file_object(fname_BM25, bm25idf)
#bm25idf = get_file_object(fname_BM25)

fname_cosines = "bm_cosines.txt"
ctx_cosines = get_cosines('mexic', vocabulary, fname_BM25)
print_cosines(fname_cosines, ctx_cosines)
#print(ctx_vectors['mexic'])