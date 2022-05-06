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

def train_value(train_data, validate_data, lr, value_save_path, policy_encoder, policy_decoder, reward_save_path, voc_len, nepoch, max_len):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    valueNetwork = ValueNetwork(voc_len).to(device)
    optimizer = optim.Adam(valueNetwork.parameters(), lr=lr)

    rewardNetwork = RewardNetwork(voc_len).to(device)
    rewardNetwork.load_state_dict(torch.load(reward_save_path, map_location=device))
    for param in rewardNetwork.parameters():
        param.requires_grad = False

    for param in policy_encoder.parameters():
        param.requires_grad = False
    for param in policy_decoder.parameters():
        param.requires_grad = False

    rewardNetwork.eval()
    policy_encoder.eval()
    policy_decoder.eval()

    criterion = nn.MSELoss().to(device)

    train_losses = []
    valid_losses = []
    print_loss = 0
    best_loss = float('inf')

    for epoch in range(nepoch):
        valueNetwork.train()
        for i, (images, captions, length) in enumerate(train_data):
            if len(images) > 1:
                images = images.to(device)
                captions = captions.to(device)

                features = policy_encoder(images)
                generated_word_ids = policy_decoder.generate(features) # generated_word_ids: (batch_size, max_len)

                ve, se = rewardNetwork(images, generated_word_ids)
                ve = F.normalize(ve, p=2, dim=1)
                se = F.normalize(se, p=2, dim=1)
                rewards = torch.sum(ve*se, axis=1).unsqueeze(1)

                values = valueNetwork(images, generated_word_ids[:,:random.randint(1, max_len)])
                loss = criterion(values, rewards)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rewardNetwork.captionRNN.hidden_state.detach_()
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
            for i, (images, captions, length) in enumerate(validate_data):
                images = images.to(device)
                captions = captions.to(device)

                features = policy_encoder(images)
                generated_word_ids = policy_decoder.generate(features)  # generated_word_ids: (batch_size, max_len)

                ve, se = rewardNetwork(images, generated_word_ids)
                ve = F.normalize(ve, p=2, dim=1)
                se = F.normalize(se, p=2, dim=1)
                rewards = torch.sum(ve * se, axis=1).unsqueeze(1)

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





