import random

import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import numpy as np
from torch.nn import functional as F
from value_model import ValueNetwork
from reward_model import RewardNetwork
from train_value_model import turnIdsToSentence
import clip

import torch.optim as optim

# ref: https://github.com/Pranshu258/Deep_Image_Captioning/blob/master/code/Deep_Captioning.ipynb
class ActorCriticNetwork(nn.Module):
    def __init__(self, valueNetwork, policy_encoder, policy_decoder):
        super(ActorCriticNetwork, self).__init__()
        self.valueNetwork = valueNetwork
        self.policy_encoder = policy_encoder
        self.policy_decoder = policy_decoder

    def forward(self, images, captions):
        with torch.no_grad():
            features = self.policy_encoder(images)
            probs = self.policy_decoder.forward_cl(features, captions)
        values = self.valueNetwork(images, captions)
        return values, probs

def buildNewData(images, captions, clip_images_features, clip_captions_features, length, level):
    # input: [1,2,3,4,...,0,0,0]
    # 需要做的事：删除尾部的0，转到前面，然后选取[:level]个
    # 一个batch中肯定至少有一个caption是没有pad的
    # length: list, cpu

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # delete the captions which is not longer than level, extract the index
    index = []
    for i in range(len(length)):
        if (length[i] > (level+1)) & (length[i] < 30):
            index.append(i)

    # get the corresponding images output
    # images: (batch_size, 3, 256, 256)
    images_output = torch.tensor([]).to(device)
    for i, image in enumerate(images):
        if i in index:
            # image: (3, 256, 256)
            image = image.unsqueeze(0)
            images_output = torch.cat((images_output, image))

    clip_images_output = torch.tensor([]).to(device)
    for i, clip_images_feature in enumerate(clip_images_features):
        if i in index:
            clip_images_feature = clip_images_feature.unsqueeze(0)
            clip_images_output = torch.cat((clip_images_output, clip_images_feature))

    clip_captions_output = torch.tensor([]).to(device)
    for i, clip_captions_feature in enumerate(clip_captions_features):
        if i in index:
            clip_captions_feature = clip_captions_feature.unsqueeze(0)
            clip_captions_output = torch.cat((clip_captions_output, clip_captions_feature))

    captions_output = torch.tensor([]).to(device)
    for i, caption in enumerate(captions):
        if i in index:
            # deal with caption, make all <pad> in caption into the front of the sentence and delete the last level characters
            # caption: tensor([   1,    4,   14,   31,    4,   40,  214,  118,    4,  621,    4,    3,
            #         1828,   31,  121,    3,   13,    2,    0,    0], device='cuda:0')
            caption_array = np.array(caption.cpu())
            real_caption_array = caption_array.ravel()[np.nonzero(caption_array)]
            num_pad = len(caption_array) - len(real_caption_array)
            final_caption = real_caption_array
            for n in range(num_pad):
                final_caption = np.insert(final_caption, 0, [0])
            final_caption = torch.tensor(final_caption).type(torch.IntTensor).to(device)
            final_caption = final_caption[:-level]
            final_caption = final_caption.unsqueeze(0)
            captions_output = torch.cat((captions_output, final_caption))

    captions_output = captions_output.type(torch.IntTensor)

    return images_output, captions_output, clip_images_output, clip_captions_output

# def buildNewData(images, captions, length, level):
#     # input: [1,2,3,4,...,0,0,0]
#     # 需要做的事：删除尾部的0，转到前面，然后选取[:level]个
#     # 一个batch中肯定至少有一个caption是没有pad的
#     # length: list, cpu

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     # delete the captions which is not longer than level, extract the index
#     index = []
#     for i in range(len(length)):
#         if (length[i] > (level+1)) & (length[i] < 30):
#             index.append(i)

#     # get the corresponding images output
#     # images: (batch_size, 3, 256, 256)
#     images_output = torch.tensor([]).to(device)
#     for i, image in enumerate(images):
#         if i in index:
#             # image: (3, 256, 256)
#             image = image.unsqueeze(0)
#             images_output = torch.cat((images_output, image))

#     captions_output = torch.tensor([]).to(device)
#     for i, caption in enumerate(captions):
#         if i in index:
#             # deal with caption, make all <pad> in caption into the front of the sentence and delete the last level characters
#             # caption: tensor([   1,    4,   14,   31,    4,   40,  214,  118,    4,  621,    4,    3,
#             #         1828,   31,  121,    3,   13,    2,    0,    0], device='cuda:0')
#             caption_array = np.array(caption.cpu())
#             real_caption_array = caption_array.ravel()[np.nonzero(caption_array)]
#             num_pad = len(caption_array) - len(real_caption_array)
#             final_caption = real_caption_array
#             for n in range(num_pad):
#                 final_caption = np.insert(final_caption, 0, [0])
#             final_caption = torch.tensor(final_caption).type(torch.IntTensor).to(device)
#             final_caption = final_caption[:-level]
#             final_caption = final_caption.unsqueeze(0)
#             captions_output = torch.cat((captions_output, final_caption))

#     captions_output = captions_output.type(torch.IntTensor)

#     return images_output, captions_output

# curriculum learning
# def monteCarlo_ac(train_data, validate_data, lr, encoder_save_path, decoder_save_path, encoder, decoder, value_save_path, voc, nepoch, max_len, log_save_path):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     voc_len = len(voc)
#     valueNetwork = ValueNetwork(voc_len).to(device)
#     valueNetwork.load_state_dict(torch.load(value_save_path, map_location=device))
#     clip_model, _ = clip.load("ViT-B/32", device=device)
#     logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(device)
#     logit_scale = logit_scale.exp()

#     acNetwork = ActorCriticNetwork(valueNetwork, encoder, decoder).to(device)
#     optimizer = optim.Adam(acNetwork.parameters(), lr=lr)

#     curriculum = [5, 9, 12]
#     # curriculum = [14]
#     # curriculum = [1, 3, 5, 7, 9, 11, 13]
#     with open(log_save_path, "w") as f:
#         for level in curriculum:
#             print_loss = 0
#             best_loss = float('inf')
#             for epoch in range(nepoch):
#                 train_losses = []
#                 valid_losses = []
#                 acNetwork.train()
#                 for i, (images, captions, length, _, _) in enumerate(train_data):
#                     if len(images)>1:
#                         values_list = torch.tensor([]).to(device)
#                         log_probs_list = torch.tensor([]).to(device)
#                         images = images.to(device)  # batch_size x
#                         captions = captions.to(device)  # batch_size x

#                         # buildNewData: make all captions are the same length and delete the last level-number word
#                         images_in, captions_in = buildNewData(images, captions, length, level)
#                         images_in = images_in.to(device)
#                         captions_in = captions_in.to(device)

#                         for j in range(level):
#                             value, probs = acNetwork(images_in, captions_in) # value: (batch_size, 1), probs: (batch_size, voc_size)
#                             # use (batch_size, voc_size) probs to compute random choice
#                             # work on cpu
#                             dist = probs.cpu().detach().numpy()
#                             actions_int = []
#                             for d in range(len(dist)):
#                                 action = np.random.choice(probs.shape[-1], p=dist[d])
#                                 actions_int.append(action)

#                             log_probs = []
#                             for a in range(len(actions_int)):
#                                 log_probs.append(torch.log(probs[a][actions_int[a]]))
#                             log_probs = torch.tensor(log_probs).unsqueeze(0).to(device)

#                             value = value.t()

#                             values_list = torch.cat((values_list, value))
#                             log_probs_list = torch.cat((log_probs_list, log_probs))

#                         loss = ((-log_probs_list) * values_list).mean()

#                         loss.requires_grad_(True)

#                         optimizer.zero_grad()
#                         loss.backward()
#                         optimizer.step()

#                         train_losses.append(loss.item())
#                         print_loss += loss.item()

#                         if i % 50 == 0:
#                             print_msg = str(level) + " [" + str(epoch + 1) + ", " + str(i + 1) + "]" + ", running_loss: " + str(
#                                 print_loss / 50)
#                             print(print_msg)
#                             f.write(print_msg + "\n")
#                             print_loss = 0.0


#                 acNetwork.eval()
#                 with torch.no_grad():
#                     for i, (images, captions, length, _, _) in enumerate(validate_data):
#                         values_list = torch.tensor([]).to(device)
#                         log_probs_list = torch.tensor([]).to(device)
#                         images = images.to(device)  # batch_size x
#                         captions = captions.to(device)  # batch_size x

#                         # buildNewData: make all captions are the same length and delete the last level-number word
#                         images_in, captions_in = buildNewData(images, captions, length, level)
#                         images_in = images_in.to(device)
#                         captions_in = captions_in.to(device)

#                         for j in range(level):
#                             value, probs = acNetwork(images_in, captions_in) # value: (batch_size, 1), probs: (batch_size, voc_size)
#                             # use (batch_size, voc_size) probs to compute random choice
#                             # work on cpu
#                             dist = probs.cpu().detach().numpy()
#                             actions_int = []
#                             for d in range(len(dist)):
#                                 action = np.random.choice(probs.shape[-1], p=dist[d])
#                                 actions_int.append(action)

#                             log_probs = []
#                             for a in range(len(actions_int)):
#                                 log_probs.append(torch.log(probs[a][actions_int[a]]))
#                             log_probs = torch.tensor(log_probs).unsqueeze(0).to(device)

#                             value = value.t()

#                             values_list = torch.cat((values_list, value))
#                             log_probs_list = torch.cat((log_probs_list, log_probs))

#                         loss = ((-log_probs_list) * values_list).mean()
#                         valid_losses.append(loss.item())

#                 train_loss = np.average(train_losses)
#                 valid_loss = np.average(valid_losses)
#                 print_msg = "level: " + str(level) +", epoch: " + str(epoch + 1) + ", train_loss: " + str(train_loss) + ", valid_loss: " + str(
#                     valid_loss)
#                 f.write(print_msg + "\n")
#                 print(print_msg)
#                 if valid_loss < best_loss:
#                     best_loss = valid_loss
#                     torch.save(encoder.state_dict(), encoder_save_path, _use_new_zipfile_serialization=False)
#                     torch.save(decoder.state_dict(), decoder_save_path, _use_new_zipfile_serialization=False)
#                 else:
#                     print("Early stopping with best_acc: ", best_loss)
#                     f.write("Early stopping with best_acc: " + str(best_loss) + "\n")
#                     break

def monteCarlo_ac(train_data, validate_data, lr, value_save_new_path, encoder, decoder, value_save_path, voc, nepoch, max_len, log_save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    voc_len = len(voc)
    valueNetwork = ValueNetwork(voc_len).to(device)
    valueNetwork.load_state_dict(torch.load(value_save_path, map_location=device))
    clip_model, _ = clip.load("ViT-B/32", device=device)
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(device)
    logit_scale = logit_scale.exp()

    acNetwork = ActorCriticNetwork(valueNetwork, encoder, decoder).to(device)
    optimizer = optim.Adam(acNetwork.parameters(), lr=lr)

    curriculum = [3, 5, 7, 9, 12]
    # curriculum = [14]
    # curriculum = [1, 3, 5, 7, 9, 11, 13]
    with open(log_save_path, "w") as f:
        for level in curriculum:
            print_loss = 0
            best_loss = float('inf')
            for epoch in range(nepoch):
                train_losses = []
                valid_losses = []
                acNetwork.train()
                for i, (images, captions, length, clip_images_features, clip_captions_features) in enumerate(train_data):
                    if len(images)>1:
                        rewards_list = torch.tensor([]).to(device)
                        values_list = torch.tensor([]).to(device)
                        log_probs_list = torch.tensor([]).to(device)
                        images = images.to(device)  # batch_size x
                        captions = captions.to(device)  # batch_size x
                        clip_images_features = clip_images_features.to(device)
                        clip_captions_features = clip_captions_features.to(device)

                        # buildNewData: make all captions are the same length and delete the last level-number word
                        images_in, captions_in, clip_images_in, clip_captions_in = buildNewData(images, captions, clip_images_features, clip_captions_features, length, level)
                        images_in = images_in.to(device)
                        captions_in = captions_in.to(device)
                        clip_images_in = clip_images_in.type(torch.FloatTensor).to(device)
                        clip_captions_in = clip_captions_in.type(torch.FloatTensor).to(device)

                        for j in range(level):
                            value, probs = acNetwork(images_in, captions_in) # value: (batch_size, 1), probs: (batch_size, voc_size)

                            # use (batch_size, voc_size) probs to compute random choice
                            # work on cpu
                            dist = probs.cpu().detach().numpy()
                            actions_int = []
                            for d in range(len(dist)):
                                action = np.random.choice(probs.shape[-1], p=dist[d])
                                actions_int.append(action)
                            # actions: [1430, 785, 361, 47, 628, 453, 879, 379]
                            actions = torch.tensor(actions_int).type(torch.IntTensor).unsqueeze(0).to(device)
                            # actions: tensor([[ 748,  541,  309, 1607,  733,  610,   63,  954]], device='cuda:0',
                            #        dtype=torch.int32)

                            # captions_in: (batch_size x sentence_length)
                            # captions_in[0]: tensor([  1,   4,  37,  39,   4,  40, 216,   7, 275, 949, 197,   4, 404, 239,
                            #          15,   3,   4, 272, 197,  18, 567,  13,   2], device='cuda:0')
                            captions_in = torch.cat((captions_in.t(),actions)).t()
                            # log_prob should be a (batch_size)
                            log_probs = []
                            for a in range(len(actions_int)):
                                log_probs.append(torch.log(probs[a][actions_int[a]]))
                            log_probs = torch.tensor(log_probs).unsqueeze(0).to(device)

                            with torch.no_grad():
                                clip_generated_captions = turnIdsToSentence(captions_in, voc)
                                if (clip_generated_captions == []):
                                    break
                                clip_generated_captions = clip_generated_captions.to(device)
                                generated_captions_features = clip_model.encode_text(clip_generated_captions).to(device)
                                generated_captions_features = generated_captions_features / generated_captions_features.norm(dim=1, keepdim=True)

                                clip_captions_in = clip_captions_in.float()
                                clip_images_in = clip_images_in.float()
                                generated_captions_features = generated_captions_features.float()

                                ref_logits_per_image = logit_scale * clip_images_in @ clip_captions_in.t()
                                logits_per_image = logit_scale * clip_images_in @ generated_captions_features.t()
                                logits_per_image = torch.diag(logits_per_image)
                                ref_logits_per_image = torch.diag(ref_logits_per_image)
                                rewards = torch.div(logits_per_image, ref_logits_per_image).unsqueeze(1).float()

                            value = value.t()
                            rewards = rewards.t()

                            rewards_list = torch.cat((rewards_list, rewards))
                            values_list = torch.cat((values_list, value))
                            log_probs_list = torch.cat((log_probs_list, log_probs))

                        advantage_list = values_list - rewards_list
                        advantage_list = torch.abs(advantage_list)
                        # actorLoss = ((-log_probs_list) * advantage_list).mean()
                        criticLoss = advantage_list.pow(2).mean()

                        loss = criticLoss
                        loss.requires_grad_(True)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        # use detach to cut down some back propagation in value network
                        acNetwork.valueNetwork.captionRNN.hidden_state[0].detach_()
                        acNetwork.valueNetwork.captionRNN.hidden_state[1].detach_()

                        train_losses.append(loss.item())
                        print_loss += loss.item()

                        if i % 50 == 0:
                            print_msg = str(level) + " [" + str(epoch + 1) + ", " + str(i + 1) + "]" + ", running_loss: " + str(
                                print_loss / 50)
                            print(print_msg)
                            f.write(print_msg + "\n")
                            print_loss = 0.0


                acNetwork.eval()
                with torch.no_grad():
                    for i, (images, captions, length, clip_images_features, clip_captions_features) in enumerate(validate_data):
                        rewards_list = torch.tensor([]).to(device)
                        values_list = torch.tensor([]).to(device)
                        log_probs_list = torch.tensor([]).to(device)
                        images = images.to(device)  # batch_size x
                        captions = captions.to(device)  # batch_size x
                        clip_images_features = clip_images_features.to(device)
                        clip_captions_features = clip_captions_features.to(device)
                        images_in, captions_in, clip_images_in, clip_captions_in = buildNewData(images, captions, clip_images_features, clip_captions_features, length, level)
                        images_in = images_in.to(device)
                        captions_in = captions_in.to(device)
                        clip_images_in = clip_images_in.type(torch.FloatTensor).to(device)
                        clip_captions_in = clip_captions_in.type(torch.FloatTensor).to(device)
                        for j in range(level):
                            value, probs = acNetwork(images_in,
                                                    captions_in)  # value: (batch_size, 1), probs: (batch_size, voc_size)

                            # use (batch_size, voc_size) probs to compute random choice
                            # work on cpu
                            dist = probs.cpu().detach().numpy()
                            actions_int = []
                            for d in range(len(dist)):
                                action = np.random.choice(probs.shape[-1], p=dist[d])
                                actions_int.append(action)
                            actions = torch.tensor(actions_int).type(torch.IntTensor).unsqueeze(0).to(device)
                            captions_in = torch.cat((captions_in.t(), actions)).t()

                            log_probs = []
                            for a in range(len(actions_int)):
                                log_probs.append(torch.log(probs[a][actions_int[a]]))
                            log_probs = torch.tensor(log_probs).unsqueeze(0).to(device)

                            # change
                            with torch.no_grad():
                                clip_generated_captions = turnIdsToSentence(captions_in, voc)
                                if (clip_generated_captions == []):
                                    break
                                clip_generated_captions = clip_generated_captions.to(device)
                                generated_captions_features = clip_model.encode_text(clip_generated_captions).to(device)
                                generated_captions_features = generated_captions_features / generated_captions_features.norm(dim=1, keepdim=True)

                                clip_captions_in = clip_captions_in.float()
                                clip_images_in = clip_images_in.float()
                                generated_captions_features = generated_captions_features.float()

                                ref_logits_per_image = logit_scale * clip_images_in @ clip_captions_in.t()
                                logits_per_image = logit_scale * clip_images_in @ generated_captions_features.t()
                                logits_per_image = torch.diag(logits_per_image)
                                ref_logits_per_image = torch.diag(ref_logits_per_image)
                                rewards = torch.div(logits_per_image, ref_logits_per_image).unsqueeze(1).float()

                            value = value.t()
                            rewards = rewards.t()

                            rewards_list = torch.cat((rewards_list, rewards))
                            values_list = torch.cat((values_list, value))
                            log_probs_list = torch.cat((log_probs_list, log_probs))

                        advantage_list = values_list - rewards_list
                        advantage_list = torch.abs(advantage_list)
                        # actorLoss = ((-log_probs_list) * advantage_list).mean()
                        criticLoss = advantage_list.pow(2).mean()

                        loss = criticLoss
                        valid_losses.append(loss.item())

                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                print_msg = "level: " + str(level) +", epoch: " + str(epoch + 1) + ", train_loss: " + str(train_loss) + ", valid_loss: " + str(
                    valid_loss)
                f.write(print_msg + "\n")
                print(print_msg)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    torch.save(valueNetwork.state_dict(), value_save_new_path, _use_new_zipfile_serialization=False)
                else:
                    print("Early stopping with best_acc: ", best_loss)
                    f.write("Early stopping with best_acc: " + str(best_loss) + "\n")
                    break