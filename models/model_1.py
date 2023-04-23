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
    
class Model1(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, num_classes, num_layers, nhead):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, num_positions, True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model*3, nhead=nhead, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.lin = nn.Linear(d_model*3, num_classes)
        self.sm = nn.Softmax(dim = 2)

        self.mask = np.zeros((num_positions, num_positions))
        for i in range(num_positions):
            for j in range(i+1):
                self.mask[i][j] = 1

    def forward(self, s, q, pa):
        # Encoding Stage:
        # Word Embed s, q, pa (word by word) -> [batch size, num_words, emb_dim]
        # Positiional Embed s, q, pa (word by word) -> [batch size, num_words, emb_dim]

        # Concat the two -> [batch_size, num_words, emb_dim * 2]
        emb_s = self.embedding(s)
        emb_q = self.embedding(q)
        emb_pa = self.embedding(pa)

        emb_s += self.pos_embedding(emb_s)
        emb_q += self.pos_embedding(emb_q)
        emb_pa += self.pos_embedding(emb_pa)

        X = torch.cat((emb_s, emb_q, emb_pa), 2)
        
        # Run through Transformer
        mask = torch.from_numpy(self.mask)
        tensor_mask = mask == 1
        output = self.transformer_encoder(X, mask = tensor_mask)     
        
        # Run Through lin layer
        lin_result = self.lin(output)

        # Softmax
        prob = self.sm(lin_result)

        # TODO what happens if we run this through another linear layer to further reduce into batch_size, 1, 2 to then get a result
        # rather than have batch_size, sent_size, 2
        return prob[:,-1,:]
