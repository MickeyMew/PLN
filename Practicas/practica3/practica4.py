from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pickle import dump, load
import nltk
from re import match
import numpy as np

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

def get_norm_tokens(clean_tokens,lemma_dict_fname):
    input = open(lemma_dict_fname,'rb')
    lemma_dict = load(input)
    input.close()
    norm_tokens = []
    for tok in clean_tokens:
        if(tok not in stopwords.words('spanish')):
            norm_tokens.append(tok)
    
    finished_tokens = []
    for tok in norm_tokens:
        if tok in lemma_dict:
            finished_tokens.append(lemma_dict[tok])
        else:
            finished_tokens.append(tok)
    return finished_tokens

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
        for x in range(0,len(vocabulary)):
             vector.append(contexts[term].count(vocabulary[x]))
        vectors[term] = vector
    return vectors

def print_context_inord(context):
    print(sorted(context.items(), key = lambda kv:(kv[1], kv[0])))

def get_vocabulary(norm_tokens):
    vocabulary = list(set(norm_tokens))
    vocabulary.sort()
    return vocabulary

def save_file_object(fname, obj):
    output = open(fname, 'wb')
    dump(obj, output, -1)
    output.close()

def get_file_object(fname):
    input = open(fname, 'rb')
    file_object = load(input)
    input.close()
    
    return file_object

def make_and_save_vectors(fname, contexts, vocabulary):
    vectors = dict()
    for term in vocabulary:
        vector = []
        for x in range(0,len(vocabulary)):
             vector.append(contexts[term].count(vocabulary[x]))
        vectors[term] = vector
    output = open(fname, 'wb')
    dump(vectors, output, -1)
    output.close()
    print(vectors)
    
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

def print_cosine(fname, cosines):
    f = open(fname, 'w')
    for item in cosines:
            f.write(str(item))
            f.write('\n')
    f.close()

'''
text_str = get_text_string("e961024.htm")
r_tokens = get_raw_tokens(text_str)
clean_tokens = get_clean_tokens(r_tokens)
fname_stem_dict = 'stem_dict.dict'
norm_tokens = get_norm_tokens(clean_tokens, fname_stem_dict)
fname_norm_tok = "stem_norm_tokens.tok"
save_file_object(fname_norm_tok, norm_tokens)
'''
fname_norm_tok = "stem_norm_tokens.tok"
norm_tokens = get_file_object(fname_norm_tok)
vocabulary = get_vocabulary(norm_tokens)
fname_contexts = "stem_contexts.ctx"
#contexts = get_contexts(norm_tokens,vocabulary)
#save_file_object(fname_contexts,contexts)    

#contexts = get_file_object(fname_contexts)
fname_vectors = "stem_vectors.vec"
#make_and_save_vectors(fname_vectors,contexts,vocabulary)

fname_cosine = "stem_cosines.txt"
print_cosine(fname_cosine,get_cosines('grande',vocabulary,fname_vectors))
