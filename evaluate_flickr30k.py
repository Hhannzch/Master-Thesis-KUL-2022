from cocoTestDataloader import cocoTestData, collate_fn
from model import Encoder
from model import Decoder
from voc_flickr30k import Voc
import torch
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms
from pycocoevalcap.eval import COCOEvalCap


def read_voc():
    voc = Voc()
    with open('data_flickr30k.txt', 'r') as f:
        for line in f.readlines():
            voc.add_word(line.split("\n")[0].split(" ")[1])
            print(line.split("\n")[0].split(" ")[0])
    return voc

# ref: https://github.com/JazzikPeng/Show-Tell-Image-Caption-in-PyTorch/blob/master/SHOW_AND_TELL_CODE_FINAL_VERSION/model_bleu.py
# ref: https://github.com/cocodataset/cocoapi/issues/343
def test(test_info, test_path, device, embed_size, hidden_size, max_length, batch_size, deterministic, num_workers, encoder_save_path, decoder_save_path):
    voc = read_voc()
    encoder = Encoder(embed_size=embed_size).eval()
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, voc_size=len(voc),
                      max_length=max_length).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.load_state_dict(torch.load(encoder_save_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_save_path, map_location=device))

    coco = COCO(test_info)
    testset = cocoTestData(coco, test_path, voc, transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()]))
    testset = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=deterministic,
                                             num_workers=num_workers, collate_fn=collate_fn)
    results = []
    for i, (images, ids) in enumerate(testset):
        images = images.to(device)
        features = encoder(images)
        generate_word_ids = decoder.generate(features)
        generate_word_ids = generate_word_ids.cpu().numpy()
        generate_captions = []


        generate_word_ids = list(generate_word_ids)
        for generate_word_id in generate_word_ids:
            generate_caption = ""
            for word_id in generate_word_id:
                word = voc.index2word[word_id]
                if word == '<end>':
                    generate_caption = generate_caption + "."
                    break
                else:
                    generate_caption = generate_caption + word
            generate_captions.append(generate_caption)

        for j in range(len(ids)):
            result = {"image_id": ids[j], "caption": generate_captions[j]}
            results.append(result)

    with open('result.txt', 'w') as f:
        for result in results:
            f.write(result + '\n')
    f.close()

    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOEvalCap(coco, coco_results)
    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, score))
