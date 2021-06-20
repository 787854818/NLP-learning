#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
from torch.nn.functional import softmax, cross_entropy

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(input_size=char_embedding_size, hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(in_features=hidden_size, out_features=len(target_vocab.char2id), bias=True)
        self.vocab_size = len(target_vocab.char2id)
        self.decoderCharEmb = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=char_embedding_size, padding_idx=target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum", ignore_index=target_vocab.char2id['<pad>'])
        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        # scores = []
        input = input.long()
        Y = self.decoderCharEmb(input)                      # (length, batch, embed_size)
        out, dec_hidden = self.charDecoder(Y, dec_hidden)   # out: (length, batch, hidden_size)
        scores = self.char_output_projection(out)           # (length, batch, vocab_size)



        # for Y_t in torch.split(Y, 1, 0):
        #     Y_t = torch.squeeze(Y_t)                            # (batch, embed_size)
        #     dec_hidden = self.charDecoder(Y_t, dec_hidden)      # A tuple of two tensors of shape (1, batch, hidden_size)
        #     h_t = torch.squeeze(dec_hidden[0], dim=0)           # (batch, hidden_size)
        #     s_t = self.char_output_projection(h_t)              # (batch, vocab_size)
        #     scores.append(s_t)
        # scores = torch.stack(scores)        # (length, batch, vocab_size)
        return scores, dec_hidden





        
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        input_sequence = char_sequence[:-1, :]                                              # (length-1, batch)
        target_sequence = char_sequence[1:, :]                                              # (length-1, batch)
        scores, dec_hidden = self.forward(input_sequence, dec_hidden)                       # socres: (length-1, batch, vocab_size)

        # possibility = softmax(scores, dim=2)                                                # (length-1, batch, vocab_size)
        # loss = torch.gather(possibility, index=input_sequence[1:, :].unsqueeze(-1), dim=-1)                 # (length-2, batch, 1)
        # loss_masks = (input_sequence[1:, :] != self.target_vocab.char2id['<pad>']).float()                  # (length-2, batch)
        # loss = loss.squeeze(-1) * loss_masks                                                                # (length-2, batch)
        # loss = loss.sum(dim=0)
        # (length-2, batch)

        loss = self.cross_entropy(scores.reshape(-1, self.vocab_size), target_sequence.reshape(-1))

        return loss




        ### END YOUR CODE

    def decode_greedy2(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        batch_size = initialStates[0].shape[1]

        current_char = torch.tensor([self.target_vocab.start_of_word for _ in range(batch_size)], dtype=torch.int).unsqueeze(0)  # (1, batch_size)
        output_word = []

        dec_hidden = initialStates
        for t in range(max_length):                         # conduct greedy search
            scores, dec_hidden = self.forward(current_char, dec_hidden)     # scores: (1, batch_size, vocab_size)
            P = softmax(scores, dim=2)                                      # (1, batch_size, vocab_size)
            _, current_char = torch.max(P, dim=2)                           # char_idx: (1, batch_size)
            output_word.append(current_char)

        output_word = torch.stack(output_word, dim=0).reshape(max_length, batch_size).permute(1, 0)     # (batch_size, max_length)
        output_word = output_word.tolist()

        for i in range(batch_size):                         # truncate outputs by the end token
            chars = output_word[i]
            try:
                end_idx = chars.index(self.target_vocab.end_of_word)
            except:
                end_idx = -1
            output_word[i] = chars[:end_idx]

        # to here, output_word is a List[List[int]]
        output_word = ["".join([self.target_vocab.id2char[char_idx] for char_idx in word]) for word in output_word]

        return output_word

        ### END YOUR CODE


    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].shape[1]
        start_token = self.target_vocab.char2id['{']
        end_token = self.target_vocab.char2id['}']
        current_char = [start_token] * batch_size
        decoder_words = ['{'] * batch_size

        current_char_tensor = torch.tensor(current_char, device = device)

        h_prev, c_prev = initialStates[0], initialStates[1]

        for t in range(max_length):

            _, (h_new, c_new) = self.forward(current_char_tensor.unsqueeze(0), (h_prev, c_prev))
            s = self.char_output_projection(h_new.squeeze(0))      # shape: (batch, self.vocab_size)
            p = torch.nn.functional.log_softmax(s, dim=1)
            current_char_tensor = torch.argmax(p, dim=1)

            for i in range(batch_size):
                decoder_words[i] += self.target_vocab.id2char[current_char_tensor[i].item()]

            h_prev = h_new
            c_prev = c_new

        for i in range(batch_size):
            decoder_words[i] = decoder_words[i][1:]
            decoder_words[i] = decoder_words[i].partition('}')[0]

        return decoder_words
