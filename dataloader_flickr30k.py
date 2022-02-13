import torch
import pandas as pd
from PIL import Image
import numpy as np
import nltk

class flickrData(torch.utils.data.Dataset):
    def __init__(self, image_path, anns_path, voc, transform = None):
        self.voc = voc
        self.transform = transform
        self.image_path = image_path
        self.anns = pd.read_table(anns_path, sep='\t', header=None)
        # print(anns[0][0]) # 1000092795.jpg#0
        # print(anns[1][0]) # Two young guys with shaggy hair look at their hands while hanging out in the yard .

    def __getitem__(self, index):
        caption_pre = str(self.anns[1][index])
        image_id = str(self.anns[0][index])
        image_id = image_id.split('#')[0]
        im = Image.open(self.image_path + '\\' + image_id)
        # process gray picture to RGB picture
        image_dim_len = len(np.array(im).shape)
        if image_dim_len == 2:
            ar = np.asarray(im, dtype=np.uint8)
            x = np.ones((ar.shape[0], ar.shape[1], 3))
            x[:, :, 0] = ar
            x[:, :, 1] = ar
            x[:, :, 2] = ar
            im = Image.fromarray(x, mode='RGB')
        if self.transform is not None:
            im = self.transform(im)

        tokens = nltk.tokenize.word_tokenize(caption_pre.lower())
        caption = []
        caption.append(self.voc('<start>'))
        for token in tokens:
            caption.append(self.voc(token))
        caption.append(self.voc('<end>'))
        caption_final = torch.Tensor(caption)
        return im, caption_final

    def __len__(self):
        return len(self.anns[1][:])

def collate_fn(data):
    # this method is used for construct mini-batch tensors from several (im, caption_final)
    # sort these data by the length of caption
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data) # unzip data
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    # padding
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths






















