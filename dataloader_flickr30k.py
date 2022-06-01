import os
from pycocotools.coco import COCO
from torchvision import transforms, utils
import torch
from dataloader_flickr30k import flickr30kData, collater, collate_fn_train
from torch.nn.utils.rnn import pack_padded_sequence
from model import Encoder, Decoder
from train import *
from voc_flickr30k import build_voc
import argparse
from evaluate_flickr30k import test
from train_reward_model import train_reward
from train_value_model import train_value
import clip
from curriculum_learning_RL import curriculumLearning_RL
from value_model import ValueNetwork
from actor_critic import monteCarlo_ac

# def getData(train_data, val_data):
#     train_data = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=args.deterministic,
#                                              num_workers=args.num_workers,
#                                              collate_fn=lambda x: collate_fn(x, preprocess))
#     val_data = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=args.deterministic,
#                                            num_workers=args.num_workers, collate_fn=lambda x: collate_fn(x, preprocess))
#     return train_data, val_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Some description.")

    parser.add_argument("--anns_path", type=str,
                        default="/root/thesis/flickr30k/annotations/annotations.token")
    parser.add_argument("--image_path", type=str,
                        default="/root/thesis/flickr30k/images")
    parser.add_argument("--images_features_path", type=str,
                        default="/root/thesis/flickr30k/images_features")
    parser.add_argument("--captions_features_path", type=str,
                        default="/root/thesis/flickr30k/captions_features")
    parser.add_argument("--result_path", type=str,
                        default="/root/thesis/results/result.json")

    parser.add_argument("--test_info", type=str,
                        default="/root/thesis/flickr30k/annotations/test_caption_coco_format_100.json")
    parser.add_argument("--test_path", type=str,
                        default="/root/thesis/flickr30k/images")
    # parser.add_argument("--test_info", type=str,
    #                     default="C:\\Users\\doris\\Downloads\\coco_val\\annotations_trainval2017\\annotations\\captions_val2017.json")
    # parser.add_argument("--test_path", type=str,
    #                     default="C:\\Users\\doris\\Downloads\\coco_val\\val2017\\val2017")
    parser.add_argument("--nmin", type=int,
                        default=50)
    parser.add_argument("--batch_size", type=int,
                        default=8)
    parser.add_argument(
        "--deterministic", action="store_false", help="Whether to shuffle the data. Default is True.",
    )
    parser.add_argument("--num_workers", type=int,
                        default=1)
    parser.add_argument("--embed_size", type=int,
                        default=256)
    parser.add_argument("--hidden_size", type=int,
                        default=512)
    parser.add_argument("--max_length", type=int,
                        default=20)
    parser.add_argument("--nepoch", type=int,
                        default=15)
    parser.add_argument("--lr", type=float,
                        default=0.001)
    parser.add_argument("--alpha", type=float,
                        default=0.7)
    parser.add_argument("--beam_size", type=int,
                        default=5)
    parser.add_argument("--candidate_range", type=int,
                        default=1)

    parser.add_argument("--encoder_save_path", type=str,
                        default="/root/thesis/s1model/encoder.pth")
    parser.add_argument("--decoder_save_path", type=str,
                        default="/root/thesis/s1model/decoder.pth")
    parser.add_argument("--reward_save_path", type=str,
                        default="/root/thesis/pre_trained_model/reward.pth")
    parser.add_argument("--value_save_path", type=str,
                        default="/root/thesis/s1model/value.pth")
    parser.add_argument("--policy_log_save_path", type=str,
                        default="/root/thesis/logs/policy_log.txt")
    parser.add_argument("--value_log_save_path", type=str,
                        default="/root/thesis/logs/value_log.txt")
    parser.add_argument("--cl_log_save_path", type=str,
                        default="/root/thesis/logs/cl_log.txt")
    parser.add_argument("--mc_log_save_path", type=str,
                        default="/root/thesis/logs/mc_log.txt")

    parser.add_argument("--encoder_save_new_path", type=str,
                        default="/root/thesis/final_model/encoder_new.pth")
    parser.add_argument("--decoder_save_new_path", type=str,
                        default="/root/thesis/final_model/decoder_new.pth")
    parser.add_argument("--value_save_new_path", type=str,
                        # default="/root/thesis/new_model/value.pth")
                        default="/root/thesis/value.pth")

    args = parser.parse_args()

    voc = build_voc(args.anns_path, args.nmin)
    # 224
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #  std=[0.229, 0.224, 0.225])
    torch.multiprocessing.set_start_method('spawn')
    dataset = flickr30kData(args.image_path, args.anns_path, args.images_features_path, args.captions_features_path,
                            voc,
                            transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                               std=[0.229, 0.224, 0.225])]))
    dataset_length = len(dataset)
    val_data_len = (int(dataset_length / 40)) * 4
    train_data_len = dataset_length - val_data_len
    print(train_data_len)
    print(val_data_len)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, preprocess = clip.load("ViT-B/32", device=device)
    train_data, val_data = torch.utils.data.random_split(dataset, [train_data_len, val_data_len])
    my_collater = collater(preprocess)

    train_data1 = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=args.deterministic,
                                              num_workers=args.num_workers, collate_fn=collate_fn_train)
    val_data1 = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=args.deterministic,
                                            num_workers=args.num_workers, collate_fn=collate_fn_train)

    train_data2 = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=args.deterministic,
                                              num_workers=args.num_workers, collate_fn=my_collater)
    val_data2 = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=args.deterministic,
                                            num_workers=args.num_workers, collate_fn=my_collater)

    encoder = Encoder(embed_size=args.embed_size).to(device)
    decoder = Decoder(embed_size=args.embed_size, hidden_size=args.hidden_size, voc_size=len(voc),
                      max_length=args.max_length).to(device)

    # torch.cuda.empty_cache()
    # train(encoder, decoder, train_data1, val_data1, device, args.lr, args.encoder_save_path, args.decoder_save_path, args.nepoch, args.policy_log_save_path)

    encoder.load_state_dict(torch.load(args.encoder_save_path, map_location=device))
    decoder.load_state_dict(torch.load(args.decoder_save_path, map_location=device))
    # train_value(train_data2, val_data2, 0.00001, args.value_save_path, encoder, decoder, voc, args.nepoch, args.max_length, args.value_log_save_path)

    # curriculumLearning_RL(train_data2, val_data2, args.lr, args.encoder_save_new_path, args.decoder_save_new_path, args.value_save_new_path,
    #   encoder, decoder, args.value_save_path, voc, args.nepoch, args.max_length, args.cl_log_save_path)
    # monteCarlo_ac(train_data2, val_data2, args.lr, args.encoder_save_new_path, args.decoder_save_new_path, encoder, decoder, args.value_save_path, voc, args.nepoch, args.max_length, args.mc_log_save_path)
    # monteCarlo_ac(train_data2, val_data2, args.lr, args.value_save_new_path,
    #   encoder, decoder, args.value_save_path, voc, args.nepoch, args.max_length, args.mc_log_save_path)

    valueNetwork = ValueNetwork(len(voc))
    valueNetwork.load_state_dict(torch.load(args.value_save_new_path, map_location=device))

    # valueNetwork.load_state_dict(torch.load(args.value_save_path, map_location=device))

    test(args.test_info, args.test_path, args.result_path, device, args.embed_size, args.hidden_size, args.max_length,
         batch_size=args.batch_size, beam_size=args.beam_size, deterministic=args.deterministic,
         num_workers=args.num_workers,
         candidate_range=args.candidate_range, alpha=args.alpha,
         #  encoder_save_path=args.encoder_save_new_path, decoder_save_path=args.decoder_save_new_path, valueNetwork=valueNetwork)
         encoder_save_path=args.encoder_save_path, decoder_save_path=args.decoder_save_path, valueNetwork=valueNetwork)
