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


def read_voc():
    voc = Voc()
    with open('data_flickr30k.txt', 'r') as f:
        for line in f.readlines():
            voc.add_word(line.split("\n")[0].split(" ")[1])
            print(line.split("\n")[0].split(" ")[0])
    return voc

# ref: https://github.com/JazzikPeng/Show-Tell-Image-Caption-in-PyTorch/blob/master/SHOW_AND_TELL_CODE_FINAL_VERSION/model_bleu.py
# ref: https://github.com/cocodataset/cocoapi/issues/343
def test(test_info, test_path, device, embed_size, hidden_size, max_length, batch_size, beam_size, deterministic, num_workers, encoder_save_path, decoder_save_path):
    voc = read_voc()
    encoder = Encoder(embed_size=embed_size).eval()
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, voc_size=len(voc),
                      max_length=max_length).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(encoder_save_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_save_path, map_location=device))

    # ref: https://github.com/jontooy/vl_demos/blob/master/Flickr8k.ipynb
    # creating coco-format dataset using flickr8k
    coco = COCO(test_info)
    testset = cocoTestData(coco, test_path, voc, transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()]))
    # testset = cocoData(coco, test_path, voc,
    #                        transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    testset = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=deterministic, num_workers=num_workers, collate_fn=collate_fn)


    results = []
    ids_helper = set()
    total = len(testset)
    for i, (images,ids) in enumerate(testset):
        for j in range(len(ids)):
            if ids[j] not in ids_helper:
                images = images.to(device)
                features = encoder(images)
                # generate_word_ids = decoder.generate(features)

                generate_word_ids = decoder.generate_beam(features, beam_size, 1, device)

                generate_word_ids = generate_word_ids.cpu().numpy()
                generate_captions = []

                # generate_word_ids = list(generate_word_ids)
                for generate_word_id in generate_word_ids:
                    generate_caption = ""
                    for word_id in generate_word_id:
                        word = voc.index2word[word_id]
                        if word == '<end>':
                            break
                        else:
                            # if ((word == '.') | (word == ',') | (word == '``')):
                            #     # generate_caption = generate_caption + word
                            #     generate_caption = generate_caption
                            if (word == '<start>') | (word == '<pad>'):
                                generate_caption = generate_caption
                            else:
                                generate_caption = generate_caption + ' ' + word
                    generate_captions.append(generate_caption)
                ids_helper.add(ids[j])
                result = {"image_id": ids[j], "caption": generate_captions[j]}
                results.append(result)
        print(str(i) + "//" + str(total))

    # should generate a json here
    jsondata = json.dumps(results)
    f = open('result.json', 'w')
    f.write(jsondata)
    f.close()

