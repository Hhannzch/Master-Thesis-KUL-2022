import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

def train(encoder, decoder, train_data, device, lr):
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # total_step = len(train_data)
    for epoch in range(5):
        for i, (images, captions, length) in enumerate(train_data):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, length, batch_first=True)[0]

            features = encoder(images)
            # features = torch.tensor(features).to(device).long()
            outputs = decoder(features, captions, length)
            # print(outputs.shape)
            # print(targets.shape)
            loss = criterion(outputs, targets)

            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = 0
            running_loss += loss.item()
            """ print training loss per 500 batch"""
            if i % 5 == 0: 
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0