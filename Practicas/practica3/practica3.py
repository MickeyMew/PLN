import nltk
from pickle import dump, load
from bs4   import BeautifulSoup
from nltk.corpus import cess_esp

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
            tagged_tokens.append(w_tagged)
    
    output = open(fname_tokens, 'wb')
    dump(tagged_tokens, output, -1)
    output.close()
    
def print_saved_tokens(fname):
    input = open(fname, 'rb')
    tagged_tokens = load(input)
    input.close()
    type(tagged_tokens)
    print(tagged_tokens)
    

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
    
    

fname = 'e961024.htm'
sentences = get_sentences(fname)

fname_tagger = 'combined_tagger.pkl'
#make_and_save_combined_tagger(fname_tagger)
fname_tokens = 'raw_tagged_tokens.tok'
#make_and_save_tagged_tokens(fname_tagger, fname_tokens, sentences)
print_saved_tokens(fname_tokens)
