import random

import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import numpy as np
from torch.nn import functional as F
from value_model import ValueNetwork
from reward_model import RewardNetwork
import clip
from PIL import Image

import torch.optim as optim

# change the generated captions into clip tokenizations form to compute the clip score
def turnIdsToSentence(generate_ids_input, voc):
    generate_ids = generate_ids_input.cpu().numpy()
    generate_captions = []
    flag = 0
    for generate_word_id in generate_ids:
        generate_caption = ""
        if (len(generate_word_id) > 75):
            flag = 1
            break
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
    if flag == 0:
        clip_captions = clip.tokenize(generate_captions)
        return clip_captions
    else:
        return []



def train_value(train_data, validate_data, lr, value_save_path, policy_encoder, policy_decoder, voc, nepoch, max_len):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    voc_len = len(voc)
    valueNetwork = ValueNetwork(voc_len).to(device)
    optimizer = optim.Adam(valueNetwork.parameters(), lr=lr)
    clip_model, _ = clip.load("ViT-B/32", device=device)

    for param in policy_encoder.parameters():
        param.requires_grad = False
    for param in policy_decoder.parameters():
        param.requires_grad = False

    policy_encoder.eval()
    policy_decoder.eval()

    criterion = nn.MSELoss().to(device)

    train_losses = []
    valid_losses = []
    print_loss = 0
    best_loss = float('inf')

    for epoch in range(nepoch):
        valueNetwork.train()
        for i, (images, captions, length, clips_image, raw_captions) in enumerate(train_data):
            if len(images) > 1:
                images = images.to(device)
                captions = captions.to(device)
                clips_image = clips_image.to(device)

                features = policy_encoder(images)
                generated_word_ids = policy_decoder.generate(features) # generated_word_ids: (batch_size, max_len)

                with torch.no_grad():
                    clip_captions = turnIdsToSentence(generated_word_ids, voc).to(device)
                    text = clip.tokenize(raw_captions).to(device)
                    logits_per_image, _ = clip_model(clips_image, clip_captions)
                    ref_logits_per_image, _ = clip_model(clips_image, text)
                    logits_per_image = torch.diag(logits_per_image)
                    ref_logits_per_image = torch.diag(ref_logits_per_image)
                    rewards = torch.div(logits_per_image, ref_logits_per_image).unsqueeze(1).float()  # to change the current Half type into float type

                values = valueNetwork(images, generated_word_ids[:,:random.randint(1, max_len)])
                loss = criterion(values, rewards)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                valueNetwork.captionRNN.hidden_state[0].detach_()
                valueNetwork.captionRNN.hidden_state[1].detach_()

                train_losses.append(loss.item())
                print_loss += loss.item()

                if i % 50 ==0:
                    print_msg = "[" + str(epoch + 1) + ", " + str(i + 1) + "]" + ", running_loss: " + str(
                        print_loss / 50)
                    print(print_msg)
                    print_loss = 0.0

        valueNetwork.eval()
        with torch.no_grad():
            for i, (images, captions, length, clips_image, raw_captions) in enumerate(validate_data):
                images = images.to(device)
                captions = captions.to(device)
                clips_image = clips_image.to(device)

                features = policy_encoder(images)
                generated_word_ids = policy_decoder.generate(features)  # generated_word_ids: (batch_size, max_len)

                clip_captions = turnIdsToSentence(generated_word_ids, voc).to(device)
                text = clip.tokenize(raw_captions).to(device)
                logits_per_image, _ = clip_model(clips_image, clip_captions)
                ref_logits_per_image, _ = clip_model(clips_image, text)
                logits_per_image = torch.diag(logits_per_image)
                ref_logits_per_image = torch.diag(ref_logits_per_image)
                rewards = torch.div(logits_per_image, ref_logits_per_image).unsqueeze(1).float()

                values = valueNetwork(images, generated_word_ids[:, :random.randint(1, max_len)])
                loss = criterion(values, rewards)
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        print_msg = "epoch: " + str(epoch + 1) + ", train_loss: " + str(train_loss) + ", valid_loss: " + str(valid_loss)
        print(print_msg)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(valueNetwork.state_dict(), value_save_path, _use_new_zipfile_serialization=False)
        else:
            print("Early stopping with best_acc: ", best_loss)
            break





