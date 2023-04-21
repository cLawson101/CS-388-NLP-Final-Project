import torch.nn as nn
import torch
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)

class Transformer(nn.Module):
    def __init__(self, vocab_index, vocab_size, num_positions, d_model, num_classes, num_layers, nhead):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()

        self.vocab_index = vocab_index
        
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_embedding = PositionalEncoding(d_model, num_positions, True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)

        self.lin = nn.Linear(d_model, num_classes)
        self.log_softmax = nn.LogSoftmax(dim = 2)

        self.mask = np.zeros((num_positions, num_positions))
        for i in range(num_positions):
            for j in range(i+1):
                self.mask[i][j] = 1

    # ASSUME INDICES ARE ALREADY ENCODED
    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        X = self.embedding(indices)

        pos_X = self.pos_embedding(X)
        X += pos_X

        output = self.transformer_encoder(X, mask = torch.from_numpy(self.mask))     

        lin_result = self.lin(output)
        softmax = self.softmax(lin_result)

        return softmax
    
class Model1(nn.Module):
    def __init__(self):
        # TODO
        print()

    def forward(self, s, q, pa):
        # Encoding Stage:
        # Word Embed s, q, pa (word by word) -> [batch size, num_words, emb_dim]
        # Positiional Embed s, q, pa (word by word) -> [batch size, num_words, emb_dim]

        # Concat the two -> [batch_size, num_words, emb_dim * 2]

        # Run through Transformer

        # Run Through lin layer

        # Softmax
        return prob