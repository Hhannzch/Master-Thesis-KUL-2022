import os
from pycocotools.coco import COCO
from torchvision import transforms, utils
import torch
from dataloader_flickr30k import flickrData, collate_fn
from model import Encoder, Decoder
from train import *
from voc_flickr30k import build_voc
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Some description.")

    parser.add_argument("--anns_path", type=str,
                        default="C:\\Users\\doris\\Downloads\\flickr30k\\annotations\\annotations.token")
    parser.add_argument("--image_path", type=str,
                        default="C:\\Users\\doris\\Downloads\\flickr30k\\images")
    parser.add_argument("--nmin", type=int,
                        default=50)
    parser.add_argument("--batch_size", type=int,
                        default=64)
    parser.add_argument(
        "--deterministic", action="store_false", help="Whether to shuffle the data. Default is True.",
    )
    parser.add_argument("--num_workers", type=int,
                        default=2)
    parser.add_argument("--embed_size", type=int,
                        default=256)
    parser.add_argument("--hidden_size", type=int,
                        default=512)
    parser.add_argument("--max_length", type=int,
                        default=15)
    parser.add_argument("--lr", type=float,
                        default=0.001)

    parser.add_argument("--encoder_save_path", type=str,
                        default="C:\\Users\\doris\\Downloads\\encoder.pth")
    parser.add_argument("--decoder_save_path", type=str,
                        default="C:\\Users\\doris\\Downloads\\decoder.pth")

    args = parser.parse_args()


    voc = build_voc(args.anns_path, args.nmin)
    dataset = flickrData(args.image_path, args.anns_path, voc,
                       transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    dataset_length = len(dataset)
    val_data_len = int(dataset_length/10)
    train_data_len = dataset_length - val_data_len
    train_data, val_data = torch.utils.data.random_split(dataset, [train_data_len, val_data_len])
    train_data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=args.deterministic,
                                             num_workers=args.num_workers, collate_fn=collate_fn)
    val_data = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=args.deterministic,
                                           num_workers=args.num_workers, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder(embed_size=args.embed_size).to(device)
    decoder = Decoder(embed_size=args.embed_size, hidden_size=args.hidden_size, voc_size=len(voc),
                      max_length=args.max_length).to(device)

    # torch.cuda.empty_cache()
    train(encoder, decoder, train_data, val_data, device, args.lr)
    torch.save(encoder.state_dict(), args.encoder_save_path, _use_new_zipfile_serialization=False)
    torch.save(decoder.state_dict(), args.decoder_save_path, _use_new_zipfile_serialization=False)