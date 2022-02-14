import nltk
from collections import Counter
import pandas as pd

class Voc():
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.initial_index = 0

    def __call__(self, word):
        if not word in self.word2index:
            return self.word2index['<unknown>']
        else:
            return self.word2index[word]

    def __len__(self):
        return len(self.word2index)

    def add_word(self, word):
        if not word in self.word2index:
            self.word2index[word] = self.initial_index
            self.index2word[self.initial_index] = word
            self.initial_index += 1

def build_voc(path, nmin): # path is annotations path
    nltk.download('punkt')
    """ read flickr30k from local path, read annotations"""
    anns = pd.read_table(path, sep='\t', header=None)
    # print(anns[0][0]) # 1000092795.jpg#0
    # print(anns[1][0]) # Two young guys with shaggy hair look at their hands while hanging out in the yard .
    anns = anns[1][:]
    captions = list(anns)
    counter = Counter()
    words = []
    for i, caption in enumerate(captions):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        if i%10000 == 0:
            print('finish %d tokenized.' % i)

    for word, count in counter.items():
        if count >= nmin:
            words.append(word)

    voc = Voc()
    fw = open('data_flickr30k.txt', 'w')
    voc.add_word('<pad>')
    fw.write('0 <pad>\n')
    voc.add_word('<start>')
    fw.write('1 <start>\n')
    voc.add_word('<end>')
    fw.write('2 <end>\n')
    voc.add_word('<unknown>')
    fw.write('3 <unknown>\n')
    for word in words:
        voc.add_word(word)
        fw.write(str(voc[word]) + ' ' + word + '\n')
    fw.close()
    return voc































