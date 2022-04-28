import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import numpy as np

class RewardEmbeddingCaptions(nn.Module):
    # embed size is for embedding layer to compress the input sentence, hidden size is the size of hidden layer in the RNN
    # hidden size is also the size of output vector, which is the similar size with the image vector
    def __init__(self, voc_size, embed_size=512, hidden_size=512):
        super(RewardEmbeddingCaptions, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.hidden_state = torch.zeros(1, 1, self.hidden_size).to(device)
        self.embed = nn.Embedding(voc_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size)

    def forward(self, captions):
        captions_embed = self.embed(captions)
        output, self.hidden_state = self.gru(captions_embed.view(len(captions_embed), 1, -1), self.hidden_state)
        return output

class RewardNetwork(nn.Module):
    def __init__(self, voc_size):
        super(RewardNetwork, self).__init__()
        self.captionRNN = RewardEmbeddingCaptions(voc_size)
        self.caption_embed = nn.Linear(512, 512)

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.image_embed = nn.Linear(resnet.fc.in_features, 512)

    def forward(self, features, captions):
        for i in range(captions.shape[1]):  # get the length of each sentences
            rewardrnn = self.captionRNN(captions[:, i])  # get the hidden state refreshed each steps, and get the final results
        rewardrnn = rewardrnn.squeeze(0).squeeze(1) # ???
        se = self.caption_embed(rewardrnn)

        with torch.no_grad():
            features = self.resnet(features)
        features = features.reshape(features.size(0), -1)
        ve = self.image_embed(features)
        return ve, se


