from cocoTestDataloader import cocoTestData, collate_fn
from model import Encoder
from model import Decoder
from voc_flickr30k import Voc
import torch
from pycocotools.coco import COCO
from torchvision import transforms
from pycocoevalcap.eval import COCOEvalCap
import json
# from dataloader import cocoData, collate_fn
from evaluate_flickr30k import read_voc

def application(application_info, application_path, device, embed_size, hidden_size, max_length, batch_size, beam_size, deterministic, num_worker, encoder_save_path, decoder_save_path):
    voc = read_voc()
    encoder = Encoder(embed_size=embed_size).eval()
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, voc_size=len(voc), max_length=max_length).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(encoder_save_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_save_path, map_location=device))

    coco = COCO(application_info)
    application_set = cocoTestData(coco, application_path, voc, transform=transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()]))
    application_set = torch.utils.data.DataLoader(application_set, batch_size=batch_size, shuffle=deterministic, num_workers=num_worker,collate_fn=collate_fn)
    results = []
    results_images = [];
    ids_helper = set()
    for i, (images, ids) in enumerate(application_set):
        images = images.to(device)
        features = encoder(images)
        generate_word_ids = decoder.generate_beam(features, beam_size, 1, device)
        generate_word_ids = generate_word_ids.cpu().numpy()
        for j in range(len(ids)):
            if ids[j] not in ids_helper:
                generate_caption = ""
                for word_id in generate_word_ids[j]:
                    word = voc.index2word[word_id]
                    if word == '<end>':
                        break
                    else:
                        if ((word == '.') | (word == ',') | (word == '`')):
                            generate_caption = generate_caption
                        elif word == '<start>':
                            generate_caption = generate_caption
                        else:
                            generate_caption = generate_caption + ' ' + word
                results.append(generate_caption)
                results_images.append(images[j].cpu())
                ids_helper.add(ids[j])

    return results_images, results

if __name__ == '__main__':
    application_info = "C:\\Users\\doris\\OneDrive\\桌面\\flickr30k_application\\application_caption_coco_format_100.json"
    application_path = "C:\\Users\\doris\\OneDrive\\桌面\\flickr30k_application\\images"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_size = 256
    hidden_size = 512
    max_length = 25
    batch_size = 1
    beam_size = 3
    deterministic = True
    num_workers = 1
    encoder_save_path = "C:\\Users\\doris\\Downloads\\encoder.pth"
    decoder_save_path = "C:\\Users\\doris\\Downloads\\decoder.pth"
    images, results = application(application_info, application_path, device, embed_size, hidden_size, max_length, batch_size,
                          beam_size, deterministic, num_workers, encoder_save_path, decoder_save_path)
    print(results)