#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_size = embed_size
        # e_char=50(参见pdf），max_word_length=21(参见utils.py)
        self.e_char = 50
        self.max_word_length = 21

        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), self.e_char, padding_idx=pad_token_idx)
        self.cnn = CNN(self.e_char, self.max_word_length, embed_size, kernel_size=5)
        self.highway = Highway(embed_size)
        self.dropout = nn.Dropout(0.3)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        sentence_length = input.shape[0]
        batch_size = input.shape[1]
        # x_emb: (sentence_length, batch_size, max_word_length, e_char)
        x_emb = self.embeddings(input)
        x_reshaped_list = list(x_emb.size())
        # 将多余的维数合并，交换第2维和第3维
        # x_reshaped: (sentence_length * batch_size, e_char, max_word_length)
        x_reshaped = x_emb.reshape(-1, x_reshaped_list[3], x_reshaped_list[2])
        # x_conv_out: (sentence_length * batch_size, embed_size)
        x_conv_out = self.cnn(x_reshaped)
        # x_conv_out: (sentence_length, batch_size, embed_size)
        # 这里要转换一下形状，为了返回x_word_emb形状与要求一致
        x_conv_out = x_conv_out.reshape(x_reshaped_list[0], x_reshaped_list[1], -1)
        x_highway = self.highway(x_conv_out)
        # x_word_emb: (sentence_length, batch_size, embed_size)
        x_word_emb = self.dropout(x_highway)
        return x_word_emb

        ### END YOUR CODE

