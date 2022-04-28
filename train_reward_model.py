import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import numpy as np
from torch.nn import functional as F
from reward_model import RewardNetwork
import torch.optim as optim

# ref: https://github.com/Pranshu258/Deep_Image_Captioning/blob/master/code/Deep_Captioning.ipynb
def RewardSemanticLoss(visuals, semantics):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    beta = 0.2
    N, D = visuals.shape
    visloss = torch.mm(visuals, semantics.t())
    visloss = visloss - torch.diag(visloss).unsqueeze(1)
    visloss = visloss + (beta / N) * (torch.ones((N, N)).to(device) - torch.eye(N).to(device))
    visloss = F.relu(visloss)
    visloss = torch.sum(visloss) / N
    semloss = torch.mm(semantics, visuals.t())
    semloss = semloss - torch.diag(semloss).unsqueeze(1)
    semloss = semloss + (beta / N) * (torch.ones((N, N)).to(device) - torch.eye(N).to(device))
    semloss = F.relu(semloss)
    semloss = torch.sum(semloss) / N
    return visloss + semloss

def train_reward(train_data, validate_data, lr, reward_save_path, voc_len, nepoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rewardNetwork = RewardNetwork(voc_len).to(device)
    optimizer = optim.Adam(rewardNetwork.parameters(), lr=lr)

    train_losses = []
    valid_losses = []
    print_loss = 0
    best_loss = float('inf')

    for epoch in range(nepoch):
        rewardNetwork.train()
        for i, (images, captions, length) in enumerate(train_data):
            if len(images) > 1:
                images = images.to(device)
                captions = captions.to(device)
                ve, se = rewardNetwork(images, captions)
                loss = RewardSemanticLoss(ve, se)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rewardNetwork.captionRNN.hidden_state.detach_()

                train_losses.append(loss.item())
                print_loss += loss.item()

                if i % 50 == 0:
                    print_msg = "[" + str(epoch + 1) + ", " + str(i + 1) + "]" + ", running_loss: " + str(
                        print_loss / 50)
                    print(print_msg)
                    print_loss = 0.0

        rewardNetwork.eval()
        with torch.no_grad():
            for i, (images, captions, length) in enumerate(validate_data):
                images = images.to(device)
                captions = captions.to(device)
                ve, se = rewardNetwork(images, captions)
                loss = RewardSemanticLoss(ve, se)
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        print_msg = "epoch: " + str(epoch + 1) + ", train_loss: " + str(train_loss) + ", valid_loss: " + str(valid_loss)
        print(print_msg)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(rewardNetwork.state_dict(), reward_save_path, _use_new_zipfile_serialization=False)
        else:
            print("Early stopping with best_acc: ", best_loss)
            break