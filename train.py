import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

def train(encoder, decoder, train_data, validate_data, device, lr, encoder_save_path, decoder_save_path, nepoch, log_save_path):
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    train_losses = []
    valid_losses = []
    avg_train_loss = []
    avg_valid_loss = []
    print_loss = 0
    best_loss = float('inf')

    # total_step = len(train_data)
    with open(log_save_path, "w") as f:
        for epoch in range(nepoch):
            encoder.train()
            decoder.train()
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

                train_losses.append(loss.item())
                print_loss += loss.item()
                """ print training loss per 500 batch"""
                if i % 5 == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, print_loss / 5))
                    print_msg = (f'[{(epoch + 1):d}, {(i + 1):d}]' +
                         f'running_loss: {(print_loss / 5):.3f} ')
                    f.write(print_msg)
                    print_loss = 0.0

            """ evaluate training result using validation dataset"""
            encoder.eval()
            decoder.eval()
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
                valid_losses.append(loss.item)

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            print_msg = (f'[{epoch:>{nepoch}}/{nepoch:>{nepoch}}] ' +
                         f'train_loss: {train_loss:.5f} ' +
                         f'valid_loss: {valid_loss:.5f}')
            print(print_msg)
            f.write(print_msg)

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(encoder.state_dict(), encoder_save_path, _use_new_zipfile_serialization=False)
                torch.save(decoder.state_dict(), decoder_save_path, _use_new_zipfile_serialization=False)
            else:
                print("Early stopping with best_acc: ", best_loss)
                f.write("Early stopping with best_acc: " + str(best_loss))
                break
    f.close()

