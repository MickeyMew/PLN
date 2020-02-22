import nltk
from pickle import dump, load
from bs4   import BeautifulSoup
from nltk.corpus import stopwords
from nltk.corpus import cess_esp
from re import match
import numpy as np

def get_sentences(fname):
    f = open(fname, encoding = 'utf-8')
    text = f.read()
    f.close()
    
    soup = BeautifulSoup(text, 'lxml')
    text_string = soup.get_text()
    
    sent_tokenizer = nltk.data.load('nltk:tokenizers/punkt/spanish.pickle')
    sentences = sent_tokenizer.tokenize(text_string)
    return sentences

def make_and_save_tagged_tokens(fname_tag, fname_tokens, sentences):
    tagged_tokens = list()
    
    input = open(fname_tag, 'rb')
    tagger = load(input)
    input.close()
    
    for s in sentences:
        tokens = nltk.word_tokenize(s)
        s_tagged = tagger.tag(tokens)
        for w_tagged in s_tagged:
            tagged_tokens.append(list(w_tagged))
    
    output = open(fname_tokens, 'wb')
    dump(tagged_tokens, output, -1)
    output.close()
    
def print_saved_tokens(fname):
    input = open(fname, 'rb')
    tagged_tokens = load(input)
    input.close()
    type(tagged_tokens)
    print(tagged_tokens)
    
def get_clean_tokens(raw_tokens):
    clean_tokens = []
    for tok in raw_tokens:
        tok[0] = tok[0].lower()
        tok[1] = tok[1].lower()
        t = ''
        for ch in tok[0]:
            if match(r'[a-záéíóúñA-ZÁÉÍÓÚÑ]', ch):
                t = t + ch
        letterToken = ''.join(t)
        if letterToken != '':
            tok[0] = t
            clean_tokens.append(tok)
    return clean_tokens;

def get_norm_tokens(clean_tokens):
    norm_tokens = []
    for tok in clean_tokens:
        if(tok[0] not in stopwords.words('spanish')):
            norm_tokens.append(tok[0] + ' ' + tok[1][0])
    return norm_tokens

def make_and_save_norm_tokens(in_fname, out_fname):
    input = open(in_fname, 'rb')
    tagged_tokens = load(input)
    input.close()
    
    clean_tokens = get_clean_tokens(tagged_tokens)
    norm_tokens = get_norm_tokens(clean_tokens)
    
    output = open(out_fname, 'wb')
    dump(norm_tokens, output, -1)
    output.close()
    print(norm_tokens)
    

def make_and_save_combined_tagger(fname):
    default_tagger = nltk.DefaultTagger('V')
    patterns=[ (r'.*o$', 'NMS'), # noun masculine singular
               (r'.*os$', 'NMP'), # noun masculine plural
               (r'.*a$', 'NFS'),  # noun feminine singular
               (r'.*as$', 'NFP')  # noun feminine singular
             ]
    regexp_tagger = nltk.RegexpTagger(patterns, backoff = default_tagger)
    #train nltk.UnigramTagger using tagged sentences from cess_esp 
    cess_tagged_sents = cess_esp.tagged_sents()
    uni_tag = nltk.UnigramTagger(cess_tagged_sents, backoff  = regexp_tagger)
    
    combined_tagger = nltk.BigramTagger(cess_tagged_sents, backoff = uni_tag)
    output = open(fname, 'wb')
    dump(combined_tagger, output, -1)
    output.close()
    
def substitute_by_lemma(lemma_fname, finished_fname, tokens_fname):
    input = open(lemma_fname, 'rb')
    lemma_dict = load(input)
    input.close()
    
    input = open(tokens_fname, 'rb')
    norm_tokens = load(input)
    input.close()
    
    finished_tokens = list()
    
    for tok in norm_tokens:
        if tok in lemma_dict:
            finished_tokens.append(lemma_dict[tok])
        else:
            finished_tokens.append(tok)
    
    output = open(finished_fname, 'wb')
    dump(finished_tokens, output, -1)
    output.close()
    
def get_context(tokens, palabra):
    context = []
    for x in range(0, len(tokens)):
        if(tokens[x] == palabra):
            for y in range(-4,4):
                if(y != 0 and (y + x >= 0) and (x + y < len(tokens))):
                    context.append(tokens[x + y])
    return context

def make_and_save_contexts(fname, finished_tokens, vocabulary):
    contexts = dict()
    for term in vocabulary:
        contexts[term] = get_context(finished_tokens, term)
    
    output = open(fname, 'wb')
    dump(contexts, output, -1)
    output.close()
    print(contexts)

def get_file_object(fname):
    input = open(fname, 'rb')
    file_object = load(input)
    input.close()
    
    return file_object

def make_and_save_vocabulary(fname, finished_tokens):
    vocabulary = list(set(finished_tokens))
    vocabulary.sort()
    
    output = open(fname, 'wb')
    dump(vocabulary, output, -1)
    output.close()
    print(vocabulary)
    
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
fname = 'e961024.htm'
sentences = get_sentences(fname)

fname_tagger = 'combined_tagger.pkl'
#make_and_save_combined_tagger(fname_tagger)
fname_tokens = 'raw_tagged_tokens.tok'
#make_and_save_tagged_tokens(fname_tagger, fname_tokens, sentences)
fname_norm_tokens = 'norm_tagged_tokens.tok'
#make_and_save_norm_tokens(fname_tokens, fname_norm_tokens)
fname_lemma_dict = 'lemma_dict.dict'
fname_finished_tokens = 'finished_tokens.tok'
#substitute_by_lemma(fname_lemma_dict, fname_finished_tokens, fname_norm_tokens)
#finished_tokens = get_file_object(fname_finished_tokens)
fname_vocabulary = 'vocabulary.voc'
#make_and_save_vocabulary(fname_vocabulary, finished_tokens)
vocabulary = get_file_object(fname_vocabulary)
fname_contexts = 'context.ctx'
#make_and_save_contexts(fname_contexts, finished_tokens, vocabulary)
#contexts = get_file_object(fname_contexts)
fname_vectors = 'vectors.vec'
cosines = get_cosines("grande a", vocabulary, fname_vectors)
print_cosine("cosine.txt", cosines)