import nltk
from collections import Counter

class Voc():
    # This class contains two dictionaries, which represent word -> index and index -> word
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

def build_voc(coco, nmin):
    nltk.download('punkt')
    captions = coco.anns.keys()
    caption_ids = list(captions)
    counter = Counter()
    words = []
    for i, id in enumerate(caption_ids):
        caption = coco.loadAnns(id)[0]['caption']
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        if i%1000 == 0:
            print("finish " + str(i) + " tokenized")
    for word,count in counter.items():
        if count >= nmin:
            words.append(word)
    voc = Voc()
    voc.add_word('<pad>')
    voc.add_word('<start>')
    voc.add_word('<end>')
    voc.add_word('<unknown>')
    for word in words:
        voc.add_word(word)
    
    return voc