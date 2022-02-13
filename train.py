import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

def train(encoder, decoder, train_data, validate_data, device, lr):
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    running_loss = 0
    validate_loss = 0
    validate_loss_compare = float('inf')
    epoch = 0

    # total_step = len(train_data)
    # for epoch in range(5):
    while (validate_loss_compare > validate_loss):
        if (validate_loss != 0.0):
            validate_loss_compare = validate_loss
        validate_loss = 0.0
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

            running_loss += loss.item()
            """ print training loss per 500 batch"""
            if i % 5 == 0: 
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

        """ evaluate training result using validation dataset"""
        for i, (images, captions, length) in enumerate(validate_data):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, length, batch_first=True)[0]

            features = encoder(images)
            # features = torch.tensor(features).to(device).long()
            outputs = decoder(features, captions, length)
            # print(outputs.shape)
            # print(targets.shape)
            loss = criterion(outputs, targets)
            validate_loss += loss.item()

        print('%.3f is the validate loss for epoch %d' % (validate_loss, epoch+1))

        epoch += 1


