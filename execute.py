import os
from pycocotools.coco import COCO
from torchvision import transforms, utils
import torch
from dataloader import cocoData, collate_fn
from model import Encoder, Decoder
from train import *
from voc import build_voc

if __name__ == '__main__':

    data = []
    with open('parameters.txt', 'r') as f:
        for line in f:
            data.append(str(line).strip('\n').split(' ')[1])
    f.close()

    # print(data)
    train_info = data[0]
    train_image = data[1]
    nmin = int(data[2])
    batch_size = int(data[3])
    shuffle = bool(data[4])
    num_workers = int(data[5])
    embed_size = int(data[6])
    hidden_size = int(data[7])
    max_length = int(data[8])
    lr = float(data[9])


    coco = COCO(train_info)

    voc = build_voc(coco, nmin)
    dataset = cocoData(coco, train_image, voc, transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()]))
    train_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size ,shuffle=shuffle , num_workers=num_workers , collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder(embed_size=embed_size).to(device)
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, voc_size=len(voc), max_length=max_length).to(device);

    train(encoder, decoder, train_data, device, lr)



# os.system('python voc.py --firstname ' + data[0] + ' --secondname '+ data[1])