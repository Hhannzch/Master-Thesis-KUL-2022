import random

import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import numpy as np
from torch.nn import functional as F
from value_model import ValueNetwork
from reward_model import RewardNetwork

import torch.optim as optim

# ref: https://github.com/Pranshu258/Deep_Image_Captioning/blob/master/code/Deep_Captioning.ipynb
class ActorCriticNetwork(nn.Module):
    def __init__(self, valueNetwork, policy_encoder, policy_decoder):
        super(ActorCriticNetwork, self).__init__()
        self.valueNetwork = valueNetwork
        self.policy_encoder = policy_encoder
        self.policy_decoder = policy_decoder

    def forward(self, images, captions, length):
        values = self.valueNetwork(images, captions)
        features = self.policy_encoder(images)
        probs = self.policy_decoder(features, captions, length)
        return values, probs

def buildNewData(images, captions, length, level):
    return images, captions

# curriculum learning
def curriculumLearning_RL(train_data, validate_data, lr, encoder_save_path, decoder_save_path, value_save_new_path, encoder, decoder, reward_save_path, value_save_path, voc_len, nepoch, max_len):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    valueNetwork = ValueNetwork(voc_len).to(device)
    valueNetwork.load_state_dict(torch.load(value_save_path, map_location=device))
    rewardNetwork = RewardNetwork(voc_len).to(device)
    rewardNetwork.load_state_dict(torch.load(reward_save_path, map_location=device))
    for param in rewardNetwork.parameters():
        param.requires_grad = False
    acNetwork = ActorCriticNetwork(valueNetwork, encoder, decoder)
    optimizer = optim.Adam(acNetwork.parameters(), lr=lr)

    curriculum = [5, 10, 15, 20, 24]

    for level in curriculum:
        print_loss = 0
        train_losses = []
        valid_losses = []
        best_lost = float('inf')
        for epoch in range(nepoch):
            acNetwork.train()
            for i, (images, captions, length) in enumerate(train_data):
                rewards = []
                values = []
                log_probs = []
                images_in, captions_in = buildNewData(images, captions, length, level)
                # batch_size == 1
                images_in.to(device)
                captions_in.to(device)
                for j in range(level):
                    value, probs = acNetwork(images_in, captions_in)
                    probs = F.softmax(probs, dim=2)
                    dist = probs.cpu().detach().numpy()[0,0]
                    action = np.random.choice(probs.shape[-1], p=dist)

                    gen_cap = torch.from_numpy(np.array([action])).unsqueeze(0).to(device)
                    captions_in = torch.cat((captions_in, gen_cap), axis=1)

                    log_prob = torch.log(probs[0, 0, action])

                    ve, se = rewardNetwork(images_in, captions_in)
                    ve = F.normalize(ve, p=2, dim=1)
                    se = F.normalize(se, p=2, dim=1)
                    reward = torch.sum(ve * se, axis=1).unsqueeze(1)


                    reward = reward.cpu().detach().numpy()[0, 0]

                    rewards.append(reward)
                    values.append(value)
                    log_probs.append(log_prob)

                values = torch.tensor(values).to(device)
                rewards = torch.tensor(rewards).to(device)
                log_probs = torch.tensor(log_probs).to(device)

                advantage = values - rewards
                actorLoss = (-log_probs * advantage).mean()
                criticLoss = 0.5 * advantage.pow(2).mean()

                loss = actorLoss + criticLoss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                print_loss += loss.item()

                if i % 50 == 0:
                    print_msg = "[" + str(epoch + 1) + ", " + str(i + 1) + "]" + ", running_loss: " + str(
                        print_loss / 50)
                    print(print_msg)
                    print_loss = 0.0

            acNetwork.eval()
            with torch.no_grad():
                for i, (images, captions, length) in enumerate(validate_data):
                    rewards = []
                    values = []
                    log_probs = []
                    images_in, captions_in = buildNewData(images, captions, length, level)
                    # batch_size == 1
                    images_in.to(device)
                    captions_in.to(device)
                    for j in range(level):
                        value, probs = acNetwork(images_in, captions_in)
                        probs = F.softmax(probs, dim=2)
                        dist = probs.cpu().detach().numpy()[0, 0]
                        action = np.random.choice(probs.shape[-1], p=dist)

                        gen_cap = torch.from_numpy(np.array([action])).unsqueeze(0).to(device)
                        captions_in = torch.cat((captions_in, gen_cap), axis=1)

                        log_prob = torch.log(probs[0, 0, action])

                        ve, se = rewardNetwork(images_in, captions_in)
                        ve = F.normalize(ve, p=2, dim=1)
                        se = F.normalize(se, p=2, dim=1)
                        reward = torch.sum(ve * se, axis=1).unsqueeze(1)

                        reward = reward.cpu().detach().numpy()[0, 0]

                        rewards.append(reward)
                        values.append(value)
                        log_probs.append(log_prob)

                    values = torch.tensor(values).to(device)
                    rewards = torch.tensor(rewards).to(device)
                    log_probs = torch.tensor(log_probs).to(device)

                    advantage = values - rewards
                    actorLoss = (-log_probs * advantage).mean()
                    criticLoss = 0.5 * advantage.pow(2).mean()

                    loss = actorLoss + criticLoss
                    valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            print_msg = "epoch: " + str(epoch + 1) + ", train_loss: " + str(train_loss) + ", valid_loss: " + str(
                valid_loss)
            print(print_msg)
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(encoder.state_dict(), encoder_save_path, _use_new_zipfile_serialization=False)
                torch.save(decoder.state_dict(), decoder_save_path, _use_new_zipfile_serialization=False)
                torch.save(valueNetwork.state_dict(), value_save_new_path, _use_new_zipfile_serialization=False)
            else:
                print("Early stopping with best_acc: ", best_loss)
                break