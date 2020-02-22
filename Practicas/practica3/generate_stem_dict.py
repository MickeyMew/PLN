import re
from pickle import dump

word_dict = dict()
f = open('generate.txt', encoding = 'latin1')
test = f.read()
f.close()

print(test)
rows = re.split("\n+", test)
rows = list(filter(None,rows))

for row in rows:
    cols = re.split(" +", row)
    cols = list(filter(None, cols))
    l_cols = len(cols)
    stem = re.split("#", cols[0])[0]
    lemma = cols[0].replace("#",'')
    word_dict[lemma] = stem 

output = open('stem_dict.dict', 'wb')
dump(word_dict, output, -1)
output.close()