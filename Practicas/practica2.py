from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
from re import match

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

def get_norm_tokens(clean_tokens):
    norm_tokens = []
    for tok in clean_tokens:
        if(tok not in stopwords.words('spanish')):
            norm_tokens.append(tok)
    return norm_tokens

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

text_str = get_text_string("e961024.htm")
r_tokens = get_raw_tokens(text_str)
clean_tokens = get_clean_tokens(r_tokens)
norm_tokens = get_norm_tokens(clean_tokens)
vocabulary = get_vocabulary(norm_tokens)
print(vocabulary)
contexts = get_contexts(norm_tokens, vocabulary)
vectors = get_vectors(contexts, vocabulary)
print(vectors['morir'])