import torch
from PIL import Image
import numpy as np
import nltk

class cocoData(torch.utils.data.Dataset):
    def __init__(self, coco, train_path, voc, transform = None):
        self.ids = list(coco.anns.keys())
        self.coco = coco
        self.train_path = train_path
        self.voc = voc
        self.transform = transform

    def __getitem__(self, index):
        id = self.ids[index]
        image_id = self.coco.loadAnns(id)[0]['image_id']
        caption_pre = str(self.coco.loadAnns(id)[0]['caption'])
        image_info = self.coco.loadImgs(image_id)
        image_path = image_info[0]['file_name']
        im = Image.open(self.train_path + '/' + image_path)
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
        # return im, caption_final, image_id
        return im, image_id

    def __len__(self):
        return len(self.ids)

# def collate_fn(data):
#     # this method is used for construct mini-batch tensors from several (im, caption_final)
#     # sort these data by the length of caption
#     data.sort(key=lambda x: len(x[1]), reverse=True)
#     images, captions, image_ids = zip(*data) # unzip data
#     images = torch.stack(images, 0)
#     lengths = [len(cap) for cap in captions]
#     targets = torch.zeros(len(captions), max(lengths)).long()
#     # padding
#     for i, cap in enumerate(captions):
#         end = lengths[i]
#         targets[i, :end] = cap[:end]
#     # return images, targets, lengths, image_ids
#     return images, image_ids

def collate_fn(data):
    # this method is used for construct mini-batch tensors from several (im, caption_final)
    # sort these data by the length of caption
    data.sort(key=lambda x: len(x), reverse=True)
    images, image_ids = zip(*data)  # unzip data
    images = torch.stack(images, 0)
    return images, image_ids



        