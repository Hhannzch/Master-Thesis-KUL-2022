import torch.nn as nn
import torchvision.models as models
import torch


class ValueNetworkEmbeddingCaptions(nn.Module):
    def __init__(self, voc_size, hidden_size, embed_size=512):
        super(ValueNetworkEmbeddingCaptions, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.hidden_state = (
        torch.zeros(1, 1, self.hidden_size).to(device), torch.zeros(1, 1, self.hidden_size).to(device))
        self.embed = nn.Embedding(voc_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)

    def forward(self, captions):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        captions = captions.type(torch.LongTensor).to(device)
        caption_embed = self.embed(captions)
        output, self.hidden_state = self.lstm(caption_embed.view(len(caption_embed), 1, -1), self.hidden_state)
        return output


class ValueNetwork(nn.Module):
    def __init__(self, voc_size, hidden_size=512):
        super(ValueNetwork, self).__init__()
        self.captionRNN = ValueNetworkEmbeddingCaptions(voc_size, hidden_size)

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear1 = nn.Linear(resnet.fc.in_features + hidden_size, 512)
        self.linear2 = nn.Linear(512, 1)

    def forward(self, features, captions):
        for i in range(captions.shape[1]):
            valuernn = self.captionRNN(captions[:, i])
        # valuernn = valuernn.squeeze(0).unsqueeze(1)
        valuernn = valuernn.squeeze(1)

        with torch.no_grad():
            features = self.resnet(features)
        features = features.reshape(features.size(0), -1)
        res = self.linear1(torch.cat((features, valuernn), dim=1))
        res = self.linear2(res)
        return res