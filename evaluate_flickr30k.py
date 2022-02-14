from model import Encoder
from model import Decoder
from voc_flickr30k import Voc
import torch
from PIL import Image
import numpy as np


def read_voc():
    voc = Voc()
    with open('data_flickr30k.txt', 'r') as f:
        for line in f.readlines():
            voc.add_word(line.split("\n")[0].split(" ")[1])
            print(line.split("\n")[0].split(" ")[0])
    return voc


def generate_caption(image_path, voc, encoder, decoder, transform, device):
    """reading image from input path"""
    im = Image.open(image_path)
    # process gray picture to RGB picture
    image_dim_len = len(np.array(im).shape)
    if image_dim_len == 2:
        ar = np.asarray(im, dtype=np.uint8)
        x = np.ones((ar.shape[0], ar.shape[1], 3))
        x[:, :, 0] = ar
        x[:, :, 1] = ar
        x[:, :, 2] = ar
        im = Image.fromarray(x, mode='RGB')
    if transform is not None:
        im = transform(im)
    image_tensor = im.to(device)

    """generate caption"""
    features = encoder(im)
    generate_word_ids = decoder.generate(features)
    generate_word_ids = generate_word_ids[0].cpu().numpy()
    generate_caption = []
    for word_id in generate_word_ids:
        word = voc.index2word[word_id]
        generate_caption.append(word)
        if word == '<end>':
            break
    return generate_caption


# ref: https://github.com/JazzikPeng/Show-Tell-Image-Caption-in-PyTorch/blob/master/SHOW_AND_TELL_CODE_FINAL_VERSION/model_bleu.py
def test(test_path, device, embed_size, hidden_size, max_length, encoder_save_path, decoder_save_path):
    voc = read_voc()
    encoder = Encoder(embed_size=embed_size).eval()
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, voc_size=len(voc),
                      max_length=max_length).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(encoder_save_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_save_path, map_location=device))

    name_caption_

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # encoder = Encoder(embed_size=args.embed_size).to(device)
    # decoder = Decoder(embed_size=args.embed_size, hidden_size=args.hidden_size, voc_size=len(voc),
    #                   max_length=args.max_length).to(device)
    #
    # # torch.cuda.empty_cache()
    # train(encoder, decoder, train_data, val_data, device, args.lr, args.encoder_save_path, args.decoder_save_path,
    #       args.nepoch, args.log_save_path)
