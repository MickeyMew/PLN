#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:05:59 2020

@author: root
"""
import nltk
from nltk.book import *
from nltk.corpus import brown
from nltk.corpus import wordnet as wn
def lexical_diversity(text):
    return len(text) / len(set(text))

def percentage(count, total):
    return 100 * count / total

fdist1 = FreqDist(text5)

fdist1.plot(1000, cumulative=False)
fdist1.plot(1000, cumulative=True)

V = set(text5)
words = [w for w in V if len(w) > 30]
print(sorted(words))

news_text = brown.words(categories='news')
romance_text = brown.words(categories='romance')
humor_text = brown.words(categories='humor')
fdist1 = nltk.FreqDist([w.lower() for w in news_text])
fdist2 = nltk.FreqDist([w.lower() for w in romance_text])
fdist3 = nltk.FreqDist([w.lower() for w in humor_text])
modals = ['love','hate','speak','control','feel','great','president']
print("\nNews\n")
for m in modals:
    print (m + ':')
    print (fdist1[m])
print("\nRomance\n")
for m in modals:
    print (m + ':')
    print (fdist2[m])
print("\nHumor\n")
for m in modals:
    print (m + ':')
    print (fdist3[m])
    
print(wn.synsets('computer'))
wn.synsets('machine')
wn.synsets('car')
wn.synsets('sandwich')
computer = wn.synset('computer.n.01')
car = wn.synset('car.n.01')
machine = wn.synset('machine.n.01')
sandwich = wn.synset('sandwich.n.01')

print(computer.path_similarity(machine))
print(car.path_similarity(machine))
print(car.path_similarity(computer))
print(computer.path_similarity(sandwich))

