import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
NEG_INF = -1e29

DEBUG = True
ENCODER = False
ATTENTION = True
HEURISTIC = True

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def mean_pooling(x, x_mask):
    mask = x_mask.unsqueeze(-1).float()
    length = torch.sum(x_mask.float(), dim=-1) # [N, ld]
    result = torch.sum(x * mask, dim=-2) / length.unsqueeze(-1)
    return result

def max_pooling(x, x_mask):
    mask = x_mask.unsqueeze(-1).float()
    x = x * mask
    result = F.max_pool2d(x, kernel_size=(x.shape[-2], 1)).squeeze(-2)
    return result

def mask_softmax(W, mask1, mask2):  # W: [N, lq, ls], mask: [N, ls]
    j_mask = joint_mask(mask1, mask2)
    W = W + (1 - j_mask.float()) * NEG_INF
    return F.softmax(W, dim=-1)

def joint_mask(mask1, mask2):
    mask1 = mask1.unsqueeze(-1)
    mask2 = mask2.unsqueeze(-2)
    mask = mask1 * mask2
    return mask

class Heuristic(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lin_r = nn.Linear(8 * hidden_size, 2 * hidden_size)
        self.lin_g = nn.Linear(8 * hidden_size, 2 * hidden_size)
        self.activation = torch.sigmoid
    
    def forward(self, x, y):
        if DEBUG and HEURISTIC:
            print("x.shape", x.shape)
            print("y.shape", y.shape)

        input = torch.cat([x, y, x*y, x-y], dim=-1)
        if DEBUG and HEURISTIC:
            print("input.shape", input.shape)

        r = gelu(self.lin_r(input))

        if DEBUG and HEURISTIC:
            print("r.shape", r.shape)

        g = self.activation(self.lin_g(input))
        if DEBUG and HEURISTIC:
            print("g.shape", g.shape)

        o = g*r + (1-g)*x

        if DEBUG and HEURISTIC:
            print("o.shape", o.shape)
        return o


class Attention(nn.Module):
    def __init__(self, dropout, self_attention, hidden_size):
        super().__init__()
        self.dropout = dropout
        self.self_attention = self_attention
        self.h = Heuristic(hidden_size)

    def forward(self, s, q, s_mask, q_mask):
        # Calculate B and C tensors
        len_s = s.shape[-2]
        q_t = torch.transpose(q, -1, -2)
        if DEBUG and ATTENTION:
            print("s.shape", s.shape)
            print("q.shape", q.shape)
            print("s_mask.shape", s_mask.shape)
            print("q_mask.shape", q_mask.shape)
            print("q_t.shape", q_t.shape)

        a = torch.matmul(s, q_t)

        if DEBUG and ATTENTION:
            print("a.shape", a.shape)

        if self.self_attention:
            diag = torch.eye(len_s).byte().unsqueeze(0)
            diag = diag.type(torch.bool)
            a = a.masked_fill(diag, NEG_INF)
        
        a_t = torch.transpose(a, -1, -2)
        b = torch.matmul(mask_softmax(a, s_mask, q_mask), q)
        c = torch.matmul(mask_softmax(a_t, q_mask, s_mask), s)

        if DEBUG and ATTENTION:
            print("a_t.shape", a_t.shape)
            print("b.shape", b.shape)
            print("c.shape", c.shape)
            print("ABOUT TO START HEURISTIC, s, b")

        # Apply special heuristic function
        s_tilde = self.h(s, b)
        if self.self_attention:
            return s_tilde
        else:
            q_tilde = self.h(q, c)
            return s_tilde, q_tilde


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size  # TODO: verify the input size
        self.pdrop = dropout
        self.n_layers = num_layers
        self.bilstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True, bidirectional=True)
        self.h0 = None
        self.c0 = None

    def forward(self, X, X_mask, dropout=True):
        H, (h0, c0) = self.bilstm(X)

        if X_mask is not None:
            H = H * X_mask.unsqueeze(-1).float()
        
        if dropout:
            H = F.dropout(H, self.pdrop, training=self.training)
        return H, (h0, c0)
    
class Model2Encoder(nn.Module):
    def __init__(self, vocab_size, char_size, d_model, hidden_size, dropout, num_layers):
        super().__init__()
        # self.lstm = BiLSTM(d_model, hidden_size*2, dropout, num_layers)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.d_model = d_model
        self.hidden_size = hidden_size
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.char_emb = nn.Embedding(char_size, d_model)
    
    def forward(self, s, s_mask, q, q_mask, s_char, q_char):
        # Preprocessing
        # create s and q char mask
        if DEBUG and ENCODER:
            print("IN MODELENCODER")

        s_char_mask = np.ones((s_char.shape[-1], s_char.shape[-1]))
        s_char_mask = torch.from_numpy(s_char_mask)
        s_char_mask = torch.tril(s_char_mask)
        s_char_mask = s_char_mask.type(torch.FloatTensor)

        q_char_mask = np.ones((q_char.shape[-1], q_char.shape[-1]))
        q_char_mask = torch.from_numpy(q_char_mask)
        q_char_mask = torch.tril(q_char_mask)
        q_char_mask = q_char_mask.type(torch.FloatTensor)
        # If have glove, insert into embedding layer

        if DEBUG and ENCODER:
            print("PRE WORD EMB")
            print("s.shape", s.shape)
            print("s_mask.shape", s_mask.shape)
            print("s_char.shape", s_char.shape)
            print("s_char_mask.shape", s_char_mask.shape)
            print("q.shape", q.shape)
            print("q_mask.shape", q_mask.shape)
            print("q_char.shape", q_char.shape)
            print("q_char_mask.shape", q_char_mask.shape)
            print()

        # Word Level Embedding
        s_word = self.word_emb(s)
        q_word = self.word_emb(q)

        if DEBUG and ENCODER:
            print("POST WORD EMB")
            print("s_word.shape", s_word.shape)
            print("q_word.shape", q_word.shape)
            print()

        # Character level Embedding
        s_char_emb = self.char_emb(s_char)
        q_char_emb = self.char_emb(q_char)

        # TODO might need more modification for s_char_emb
        # s_char_emb = s_char_emb.contiguous().reshape(-1, lw, self.d_model)
        # q_char_emb = q_char_emb.contiguous().reshape(-1, lw, self.d_model)
        
        # s_char_mask = s_char_mask.contiguous().reshape(-1, lw)
        # q_char_mask = q_char_mask.contiguous().reshape(-1, lw)
        # TODO might need more modification for q_char_emb

        if DEBUG and ENCODER:
            print("POST CHAR EMB")
            print("s_char_emb.shape", s_char_emb.shape)
            print("s_char_mask.shape", s_char_mask.shape)
            print("q_char_emb.shape", q_char_emb.shape)
            print("q_char_mask.shape", q_char_mask.shape)
            print()

        # Character level BiLSTM
        # _H, (s_char, _c) = self.lstm(s_char_emb, s_char_mask)
        s_char = self.transformer_encoder(s_char_emb, mask = s_char_mask)

        if DEBUG and ENCODER:
            print("POST Transformer")
            print("s_char", s_char.shape)

        s_char = s_char.contiguous().reshape(s_word.shape[0], s_word.shape[1],-1)

        if DEBUG and ENCODER:
            print("POST reshape")
            print("s_char", s_char.shape)

        # _H, (q_char, _c) = self.lstm(q_char_emb, q_char_mask)
        q_char = self.transformer_encoder(q_char_emb, mask = q_char_mask)

        if DEBUG and ENCODER:
            print("POST Transformer")
            print("q_char", q_char.shape)

        q_char = q_char.contiguous().reshape(q_word.shape[0], q_word.shape[1],-1)

        if DEBUG and ENCODER:
            print("POST reshape")
            print("q_char", q_char.shape)

        # Concat

        if DEBUG and ENCODER:
            print("PRE CAT")
            print("s_word", s_word.shape)
            print("s_char", s_char.shape)

        s_tensor = torch.cat([s_word, s_char], axis = -1)
        q_tensor = torch.cat([q_word, q_char], axis = -1)
        return s_tensor, q_tensor

class Model2(nn.Module):
    def __init__(self, vocab_size, char_size, d_model, hidden_size, dropout, num_layers):
        super().__init__()
        # ENCODE
        self.encode = Model2Encoder(vocab_size, 
                                    char_size, 
                                    d_model, 
                                    hidden_size,
                                    dropout=dropout,
                                    num_layers=num_layers)

        # First Transformer
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=d_model + hidden_size, nhead=1, batch_first = True)
        self.transformer_encoder_1 = nn.TransformerEncoder(self.encoder_layer_1, num_layers=num_layers)

        # First Attention Layer
        self.attention_start = Attention(dropout, False, hidden_size)

        # Second BiLSTMs
        self.lstm2_s = BiLSTM(hidden_size * 2, hidden_size, dropout, num_layers)
        self.lstm2_q = BiLSTM(hidden_size * 2, hidden_size, dropout, num_layers)
        # Second Transfomer
        self.encoder_layer_2_s = nn.TransformerEncoderLayer(d_model=d_model + hidden_size, nhead=1, batch_first = True)
        self.transformer_encoder_2_s = nn.TransformerEncoder(self.encoder_layer_2_s, num_layers=num_layers)

        self.encoder_layer_2_q = nn.TransformerEncoderLayer(d_model=d_model + hidden_size, nhead=1, batch_first = True)
        self.transformer_encoder_2_q = nn.TransformerEncoder(self.encoder_layer_2_q, num_layers=num_layers)


        # Second Attention Layer
        self.attention_s = Attention(dropout, True, hidden_size)
        self.attention_q = Attention(dropout, True, hidden_size)

        # Third BiLSTMs
        self.encoder_layer_3_s = nn.TransformerEncoderLayer(d_model=2*(d_model + hidden_size), nhead=1, batch_first = True)
        self.transformer_encoder_3_s = nn.TransformerEncoder(self.encoder_layer_3_s, num_layers=num_layers)

        self.encoder_layer_3_q = nn.TransformerEncoderLayer(d_model=2*(d_model + hidden_size), nhead=1, batch_first = True)
        self.transformer_encoder_3_q = nn.TransformerEncoder(self.encoder_layer_3_q, num_layers=num_layers)

        # FFN
        self.linear = nn.Linear(hidden_size * 8, 2)

        self.sm = nn.Softmax(dim = -1)

    def forward(self, s, q, s_char, q_char):
        # TODO figure out mask situation
        # TODO Mask is 1 if the value doesn't exist, and 0 if it does
        s_mask = np.ones((s.shape[-1], s.shape[-1]))
        s_mask = torch.from_numpy(s_mask)
        s_mask = torch.tril(s_mask)
        s_mask = s_mask.type(torch.FloatTensor)

        # TODO TEST THIS
        q_mask = np.ones((q.shape[-1], q.shape[-1]))
        q_mask = torch.from_numpy(q_mask)
        q_mask = torch.tril(q_mask)
        q_mask = q_mask.type(torch.FloatTensor)

        # I.e. if sentence is too short, 1's will be in the place where the sentence does
        # Not exist

        if DEBUG:
            print("PRE EMBED")
            print("s.shape", s.shape)
            print("q.shape", q.shape)
            print("s_char.shape", s_char.shape)
            print("q_char.shape", q_char.shape)
            print()

        # Encoding
        s_emb, q_emb = self.encode(s, s_mask, q, q_mask, s_char, q_char)

        if DEBUG:
            print("POST ENCODE")
            print("s_emb.shape", s_emb.shape)
            print("q_emb.shape", q_emb.shape)
            print()

        # First Transformer
        tran_1_s = self.transformer_encoder_1(s_emb, mask = s_mask)
        tran_1_q = self.transformer_encoder_1(q_emb, mask = q_mask)

        if DEBUG:
            print("POST TRANSFORMER")
            print("tran_1_s.shape", tran_1_s.shape)
            print("tran_1_q.shape", tran_1_q.shape)

        # Inference
        s_tilde, q_tilde = self.attention_start(tran_1_s, tran_1_q, s_mask, q_mask)

        if DEBUG:
            print("POST attention")
            print("s_tilde.shape", s_tilde.shape)
            print("q_tilde.shape", q_tilde.shape)
        
        # Intra-sentence Modeling
        tran_2_s = self.transformer_encoder_2_s(s_tilde, mask = s_mask)
        tran_2_q = self.transformer_encoder_2_q(q_tilde, mask = q_mask)

        if DEBUG:
            print("POST LSTM 2")
            print("tran_2_s.shape", tran_2_s.shape)
            print("tran_2_q.shape", tran_2_q.shape)

        # Intra-sentence self attention
        s_hat = self.attention_s(tran_2_s, tran_2_s, s_mask, q_mask)
        q_hat = self.attention_q(tran_2_q, tran_2_q, q_mask, s_mask)

        if DEBUG:
            print("POST self-attention")
            print("s_hat.shape", s_hat.shape)
            print("q_hat.shape", q_hat.shape)

        # Residual connection
        res_s = torch.cat([s_tilde, s_hat], dim=-1)
        res_q = torch.cat([q_tilde, q_hat], dim=-1)

        if DEBUG:
            print("POST CAT")
            print("res_s.shape", res_s.shape)
            print("res_q.shape", res_q.shape)

        # Prediction
        s_bar = self.transformer_encoder_3_s(res_s, mask = s_mask)
        q_bar = self.transformer_encoder_3_q(res_q, mask = q_mask)

        if DEBUG:
            print("POST Transformer 3")
            print("s_bar.shape", s_bar.shape)
            print("q_bar.shape", q_bar.shape)

        # Mean-max pooling
        s_mean = mean_pooling(s_bar, s_mask)
        q_mean = mean_pooling(q_bar, q_mask)

        if DEBUG:
            print("POST MEAN")
            print("s_mean.shape", s_mean.shape)
            print("q_mean.shape", q_mean.shape)

        s_max = max_pooling(s_bar, s_mask)
        q_max = max_pooling(q_bar, q_mask)

        if DEBUG:
            print("POST MAX")
            print("s_max.shape", s_max.shape)
            print("q_max.shape", q_max.shape)

        s_mm = s_mean + s_max
        q_mm = q_mean + q_max

        # mean-max concat
        s_q_concat = torch.cat([s_mm, q_mm], dim = -1)

        if DEBUG:
            print("POST CAT")
            print("s_q_concat.shape", s_q_concat.shape)

        # FFN
        lin_res = self.linear(s_q_concat)

        if DEBUG:
            print("POST LIN")
            print("lin_res.shape", lin_res.shape)

        # Softmax
        probs = self.sm(lin_res)

        if DEBUG:
            print("POST SOFTMAX")
            print("probs.shape", probs.shape)
        
        return probs