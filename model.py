import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch

class Encoder(nn.Module):
  def __init__(self, embed_size):
    super(Encoder, self).__init__()


  def forward(self, images):
      # TODO

class Decoder(nn.Module):
  def __init__(self, embed_size, hidden_size, voc_size, max_length):
    super(Decoder,self).__init__()


  def forward(self, features, captions, lengths):
      # TODO


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

  # ref: https://github.com/Moeinh77/Image-Captioning-with-Beam-Search/blob/master/main.ipynb
  def generate_beam(self, features, beam_size, start_index, device):
    start_word = torch.tensor([[start_index, 0.0]]).to(device)
    while list(start_word[0][:-1].size())[0] < self.max_length:
        inputs = features.unsqueeze(1)
        states = None
        hiddens, states = self.lstm(inputs, states)
        temp = torch.tensor([]).to(device)


        # start_word: tensor([[1.0000, -1.0119],
        #         [2.0000, 3.3243],
        #         [3.0000, 4.4325]], device='cuda:0')
        for i,(s) in enumerate(start_word):
            # s: tensor([1.,0.], device='cuda:0')
            predicted = s[:-1] # predicted: tensor([1.]), all labels except the last probability
            for p in predicted: # p: tensor(1) or tensor(33), the label of generated caption
                p = p.type(torch.IntTensor).to(device).unsqueeze(0)
                inputs = self.embed(p)
                inputs = inputs.unsqueeze(1)
                hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            softmax = torch.nn.Softmax(dim=1)
            outputs_prob = softmax(outputs)
            top_probs, top_labels = torch.topk(outputs_prob, beam_size) # top_preds.size(): batch_size x beam_size, top_labs.size(): batch_size x beam_size
            # note: here the labels is the index in the outputs instead of voc_index
            # make batch_size is 1

            # top_probs: tensor([[0.6566, 0.0589]], device='cuda:0', grad_fn=<TopkBackward0>)
            # top_label: tensor([[33, 72]], device='cuda:0')
            # s: tensor([1., 0.], device='cuda:0')
            top_probs = top_probs[0] # tensor([0.6566, 0.0589], device='cuda:0', grad_fn=<SelectBackward0>)
            top_labels = top_labels[0] # tensor([33, 72], device='cuda:0')

            for i, (pre) in enumerate(top_probs):
                next_cap, prob = s[:-1], s[-1]
                # next_cap: tensor([1.], device='cuda:0')
                # prob: tensor(0., device='cuda:0')
                next_word = top_labels[i] # tensor(33, device='cuda:0')
                next_word = next_word.unsqueeze(0) # tensor([33], device='cuda:0')
                next_cap = torch.cat((next_cap,next_word))
                # prob: tensor(0., device='cuda:0')
                # pre: tensor(0.6566, device='cuda:0', grad_fn=<UnbindBackward0>)
                pre = torch.tensor(pre.item()).to(device)
                prob += pre
                # next_cap： tensor([ 1., 33.], device='cuda:0')
                # prob: tensor(0.6566, device='cuda:0')
                prob = prob.unsqueeze(0) # tensor([0.6566], device='cuda:0')
                new = torch.cat((next_cap, prob))
                new = new.unsqueeze(0)
                temp = torch.cat((temp, new))
        start_word = temp
        start_word = sorted(start_word, reverse=True, key=lambda l: l[-1])
        start_word = start_word[:beam_size]

    start_word = start_word[0][:-1]
    return start_word.unsqueeze(0)