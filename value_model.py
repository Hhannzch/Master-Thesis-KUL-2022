import torch.nn as nn
import torchvision.models as models
import torch

class ValueNetworkEmbeddingCaptions(nn.Module):
    def __init__(self, voc_size, embed_size=512, hidden_size=512):
        super(ValueNetworkEmbeddingCaptions, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_size = embed_size
        self.hidden_size = hidden_size
