import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

def train(encoder, decoder, train_data, validate_data, device, encoder_lr, decoder_lr, encoder_save_path, decoder_save_path, nepoch, log_save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    # params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=encoder_lr)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
    train_losses = []
    valid_losses = []
    print_loss = 0
    best_loss = float('inf')
    torch.cuda.empty_cache()

    # total_step = len(train_data)
    with open(log_save_path, "w") as f:
        for epoch in range(nepoch):
            encoder.train()
            decoder.train()
            for i, (images, captions, length) in enumerate(train_data):
                images = images.to(device)
                captions = captions.to(device)
                # targets = pack_padded_sequence(captions, length, batch_first=True)[0].to(device)

                features = encoder(images)
                # features = torch.tensor(features).to(device).long()
                tensorlength = torch.tensor(length).to(device)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, captions, tensorlength)
                targets = caps_sorted[:, 1:]
                scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

                # Calculate loss
                loss = criterion(scores, targets)
                alpha_c = 1
                # Add doubly stochastic attention regularization
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                train_losses.append(loss.item())
                print_loss += loss.item()
                """ print training loss per 500 batch"""
                if i % 50 == 0:
                    print_msg = "[" + str(epoch+1) + ", "+ str(i+1) + "]" + ", running_loss: " + str(print_loss/50)
                    print(print_msg)
                    f.write(print_msg + "\n")
                    print_loss = 0.0

            """ evaluate training result using validation dataset"""
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                for i, (images, captions, length) in enumerate(validate_data):
                    images = images.to(device)
                    captions = captions.to(device)
                    # targets = pack_padded_sequence(captions, length, batch_first=True)[0]

                    features = encoder(images)
                    # features = torch.tensor(features).to(device).long()
                    tensorlength = torch.tensor(length).to(device)
                    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, captions, tensorlength)
                    targets = caps_sorted[:, 1:]
                    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
                    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

                    # Calculate loss
                    loss = criterion(scores, targets)
                    alpha_c = 1
                    # Add doubly stochastic attention regularization
                    loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
                    valid_losses.append(loss.item())

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            print_msg = "epoch: " + str(epoch+1) + ", train_loss: " + str(train_loss) + ", valid_loss: " + str(valid_loss)
            print(print_msg)
            f.write(print_msg + "\n")

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(encoder.state_dict(), encoder_save_path, _use_new_zipfile_serialization=False)
                torch.save(decoder.state_dict(), decoder_save_path, _use_new_zipfile_serialization=False)
            else:
                print("Early stopping with best_acc: ", best_loss)
                f.write("Early stopping with best_acc: " + str(best_loss) + "\n")
                break
    f.close()