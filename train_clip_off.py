import torch
import pandas as pd
import clip
from PIL import Image
import numpy as np

# model from clip github: model(image, text)
# def forward(self, image, text):
#         image_features = self.encode_image(image)
#         text_features = self.encode_text(text)

#         # normalized features
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ text_features.t()
#         logits_per_text = logits_per_image.t()

#         # shape = [global_batch_size, global_batch_size]
#         return logits_per_image, logits_per_text


class clipData(torch.utils.data.Dataset):
    def __init__(self, image_path, anns_path):
        self.image_path = image_path
        self.anns = pd.read_table(anns_path, sep='\t', header=None)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.anns[1][:])

    def __getitem__(self, index):
        image_id_out = str(self.anns[0][index])
        image_id = image_id_out.split("#")[0]
        im = Image.open(self.image_path + "/" + image_id)
        caption = str(self.anns[1][index])

        caption_words = caption.split(" ")
        if (len(caption_words)>50):
            str_help = " "
            temp = str_help.join(caption_words[:30])
            caption = temp

        return image_id_out, im, caption

class collater():
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess

    def __call__(self, data):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        image_ids, raw_images, raw_captions = zip(*data)

        clip_images = torch.stack([self.preprocess(raw_image) for raw_image in raw_images], 0).to(device)
        images_features = self.model.encode_image(clip_images)
        images_features = images_features / images_features.norm(dim=1, keepdim=True)

        clip_texts = clip.tokenize(raw_captions).to(device)
        texts_features = self.model.encode_text(clip_texts)
        texts_features = texts_features / texts_features.norm(dim=1, keepdim=True)

        return list(image_ids), images_features, texts_features


# save the models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_path = "/root/thesis/flickr30k/images"
anns_path = "/root/thesis/flickr30k/annotations/annotations.token"

model, preprocess = clip.load("ViT-B/32", device=device)
collate_fn = collater(model, preprocess)


dataset = clipData(image_path, anns_path)
clipData = torch.utils.data.DataLoader(dataset, batch_size=64, collate_fn=collate_fn)
origin = torch.tensor([])
datalen = len(clipData)

for i, (ids, images, captions) in enumerate(clipData):
    save_images_path = "/root/thesis/flickr30k/images_features/"
    save_captions_path = "/root/thesis/flickr30k/captions_features/"
    for j in range(len(ids)):
        save_image_path = save_images_path + str(ids[j]) + ".pt"
        torch.save(images[j], save_image_path)

        save_caption_path = save_captions_path + str(ids[j]) + ".pt"
        torch.save(captions[j], save_caption_path)

    if i%100 == 0:
        print("finish: " + str(i) + " of " + str(datalen))

# test

# with torch.no_grad():
#     test_image = torch.load("/root/thesis/flickr30k/images_features/2330069984.jpg#0.pt")
# print(test_image.size())
# test_caption = torch.load("/root/thesis/flickr30k/captions_features/2330069984.jpg#0.pt")
# print(test_caption.size())
# print(test_image.type)
# print(test_caption.type)