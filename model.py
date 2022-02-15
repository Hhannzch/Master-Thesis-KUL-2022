import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch

class Encoder(nn.Module):
  def __init__(self, embed_size):
    super(Encoder, self).__init__()
    resnet = models.resnet152(pretrained=True)
    modules = list(resnet.children())[:-1]
    self.resnet = nn.Sequential(*modules)
    # print(resnet.fc.in_features)
    self.linear = nn.Linear(resnet.fc.in_features, embed_size)
    self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

  def forward(self, images):
    with torch.no_grad():
      features = self.resnet(images)
    features = features.reshape(features.size(0), -1)
    features = self.bn(self.linear(features))
    # features = self.linear(features)
    # print(features)
    return features

class Decoder(nn.Module): 
  def __init__(self, embed_size, hidden_size, voc_size, max_length):
    super(Decoder,self).__init__()
    self.embed = nn.Embedding(voc_size, embed_size)
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, batch_first=True)
    self.linear = nn.Linear(hidden_size, voc_size)
    self.max_length = max_length

  def forward(self, features, captions, lengths):
    embeddings = self.embed(captions)
    embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
    packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
    hiddens, _ = self.lstm(packed)
    outputs = self.linear(hiddens[0])
    # print(outputs.shape)
    return outputs

  # generate function: https://github.com/JazzikPeng/Show-Tell-Image-Caption-in-PyTorch/blob/master/SHOW_AND_TELL_CODE_FINAL_VERSION/model_bleu.py
  def generate(self, features, states=None):
    """Generate captions for given image features using greedy search."""
    generate_word_ids = []
    inputs = features.unsqueeze(1)
    for i in range(self.max_length):
        hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
        outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
        _, predicted = outputs.max(1)                        # predicted: (batch_size)
        generate_word_ids.append(predicted)
        inputs = self.embed(predicted)
        inputs = inputs.unsqueeze(1)
    generate_word_ids = torch.stack(generate_word_ids, 1)
    return generate_word_ids
