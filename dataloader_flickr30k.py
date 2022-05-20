import torch
import pandas as pd
from PIL import Image
import numpy as np
import nltk

class flickr30kData(torch.utils.data.Dataset):
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
        im_out = im
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
        return im, caption_final, caption_pre, im_out

    def __len__(self):
        return len(self.anns[1][:])


class collater():
    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, data):
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions, raw_captions, raw_images = zip(*data) # unzip data
        images = torch.stack(images, 0)
        lengths = [len(cap) for cap in captions]

        clip_images = torch.stack([self.preprocess(raw_image) for raw_image in raw_images], 0)

        targets = torch.zeros(len(captions), max(lengths)).long()
        # padding
        for i, cap in enumerate(captions):
            if lengths[i] >= 76:
                lengths[i] = 30
                change_raw_caption = raw_captions[i]
                change_raw_captions = change_raw_caption.split(" ")[:30]
                str = " "
                temp = str.join(change_raw_captions)
                print(temp)

                raw_captions = list(raw_captions)
                raw_captions[i] = temp
                raw_captions = tuple(raw_captions)

                print(raw_captions[i])


            end = lengths[i]
            targets[i, :end] = cap[:end]

        return images, targets, lengths, clip_images, raw_captions

def collate_fn_train(data):
    # this method is used for construct mini-batch tensors from several (im, caption_final)
    # sort these data by the length of caption
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, _, _ = zip(*data) # unzip data
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    # padding
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths



# def collate_fn(data, preprocess):
#     # this method is used for construct mini-batch tensors from several (im, caption_final)
#     # sort these data by the length of caption
#
#     data.sort(key=lambda x: len(x[1]), reverse=True)
#     images, captions, raw_captions, raw_images = zip(*data) # unzip data
#     images = torch.stack(images, 0)
#     lengths = [len(cap) for cap in captions]
#
#     clip_images = torch.stack([preprocess(raw_image) for raw_image in raw_images], 0)
#
#     targets = torch.zeros(len(captions), max(lengths)).long()
#     # padding
#     for i, cap in enumerate(captions):
#         if lengths[i] >= 76:
#             lengths[i] = 76
#             change_raw_caption = raw_captions[i]
#             change_raw_captions = change_raw_caption.split(" ")[:76]
#             str = " "
#             # raw_captions = list(raw_captions)
#             raw_captions[i] = str.join(change_raw_captions);
#             # raw_captions = tuple(raw_captions)
#
#
#         end = lengths[i]
#         targets[i, :end] = cap[:end]
#
#
#
#
#     return images, targets, lengths, clip_images, raw_captions






















