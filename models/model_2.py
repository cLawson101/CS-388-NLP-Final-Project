import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
NEG_INF = -1e29


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
        input = torch.cat([x, y, x*y, x-y], dim=-1)
        r = gelu(self.lin_r(input))
        g = self.activation(self.lin_g(input))
        o = g*r + (1-g)*x
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
        a = torch.matmul(s, q_t)

        if self.self_attention:
            diag = torch.eye(len_s).byte().unsqueeze(0)
            a.data.mask_fill_(diag, NEG_INF)
        
        a_t = torch.transpose(a, -1, -2)
        b = torch.matmul(mask_softmax(a, s_mask, q_mask), q)
        c = torch.matmul(mask_softmax(a_t, q_mask, s_mask), s)

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
        self.lstm = BiLSTM(d_model, hidden_size*2, dropout, num_layers)

        self.d_model = d_model
        self.hidden_size = hidden_size
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.char_emb = nn.Embedding(char_size, d_model)
    
    def forward(self, s, s_mask, q, q_mask, s_char, q_char):
        # Preprocessing
        # create s and q char mask
        s_char_mask = s_char.clone()
        s_char_mask[s_char_mask != 0] = -1
        s_char_mask[s_char_mask == 0] = 1
        s_char_mask[s_char_mask == -1] = 0

        q_char_mask = q_char.clone()
        q_char_mask[q_char_mask != 0] = -1
        q_char_mask[q_char_mask == 0] = 1
        q_char_mask[q_char_mask == -1] = 0
        # If have glove, insert into embedding layer

        # Word Level Embedding
        s_word = self.word_emb(s)
        q_word = self.word_emb(q)

        # Character level Embedding
        s_char_emb = self.char_emb(s_char)
        q_char_emb = self.char_emb(q_char)

        bs = 128
        ld = 150
        lq = 150
        lw = 40

        # TODO might need more modification for s_char_emb
        s_char_emb = s_char_emb.contiguous().reshape(-1, lw, self.d_model)
        q_char_emb = q_char_emb.contiguous().reshape(-1, lw, self.d_model)
        
        s_char_mask = s_char_mask.contiguous().reshape(-1, lw)
        q_char_mask = q_char_mask.contiguous().reshape(-1, lw)
        # TODO might need more modification for q_char_emb

        
        print("s_word.shape", s_word.shape)
        print("s_char_emb.shape", s_char_emb.shape)

        # Character level BiLSTM
        _H, (s_char, _c) = self.lstm(s_char_emb, s_char_mask)
        # TODO maybe more modification here
        s_char = s_char.contiguous().reshape(bs, ld, 2*self.hidden_size)
        _H, (q_char, _c) = self.lstm(q_char_emb, q_char_mask)
        # TODO maybe more modification here
        q_char = q_char.contiguous().reshape(bs, ld, 2*self.hidden_size)

        # Concat
        print("s_word.shape", s_word.shape)
        print("s_char.shape", s_char.shape)
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

        # First BiLSTMs
        self.lstm1 = BiLSTM(d_model + 2* hidden_size, hidden_size, dropout, num_layers)

        # First Attention Layer
        self.attention_start = Attention(dropout, False, hidden_size)

        # Second BiLSTMs
        self.lstm2_s = BiLSTM(hidden_size * 2, hidden_size, dropout, num_layers)
        self.lstm2_q = BiLSTM(hidden_size * 2, hidden_size, dropout, num_layers)
        
        # Second Attention Layer
        self.attention_s = Attention(dropout, True, hidden_size)
        self.attention_q = Attention(dropout, True, hidden_size)

        # Third BiLSTMs
        self.lstm3_s = BiLSTM(hidden_size * 4, hidden_size, dropout, num_layers)
        self.lstm3_q = BiLSTM(hidden_size * 4, hidden_size, dropout, num_layers)
        # FFN
        self.linear = nn.Linear(hidden_size * 4, 2)

        self.sm = nn.Softmax(dim = -1)

    def forward(self, s, q, s_char, q_char):
        # TODO figure out mask situation
        # TODO Mask is 1 if the value doesn't exist, and 0 if it does
        s_mask = s.clone()
        s_mask[s_mask != 0] = -1
        s_mask[s_mask == 0] = 1
        s_mask[s_mask == -1] = 0

        # TODO TEST THIS
        q_mask = q.clone()
        q_mask[q_mask != 0] = -1
        q_mask[q_mask == 0] = 1
        q_mask[q_mask == -1] = 0

        # I.e. if sentence is too short, 1's will be in the place where the sentence does
        # Not exist

        # Encoding
        s_emb, q_emb = self.encode(s, s_mask, q, q_mask, s_char, q_char)

        # First BiLSTM
        lstm_1_s = self.lstm1(s_emb, s_mask)
        lstm_1_q = self.lstm1(q_emb, q_mask)

        # Inference
        s_tilde, q_tilde = self.attention_start(lstm_1_s, lstm_1_q, s_mask, q_mask)

        # Intra-sentence Modeling
        lstm_2_s, _ = self.lstm2_s(s_tilde, s_mask)
        lstm_2_q, _ = self.lstm2_q(q_tilde, q_mask)

        # Intra-sentence self attention
        s_hat = self.attention_s(lstm_2_s, lstm_2_s, s_mask)
        q_hat = self.attention_q(lstm_2_q, lstm_2_q, q_mask)

        # Residual connection
        res_s = torch.cat([s_tilde, s_hat], dim=-1)
        res_q = torch.cat([q_tilde, q_hat], dim=-1)

        # Prediction
        s_bar, _ = self.lstm3_s(res_s, s_mask)
        q_bar, _ = self.lstm3_q(res_q, q_mask)

        # Mean-max pooling
        s_mean = mean_pooling(s_bar, s_mask)
        q_mean = mean_pooling(q_bar, q_mask)

        s_max = max_pooling(s_bar, s_mask)
        q_max = max_pooling(q_bar, q_mask)

        s_mm = s_mean + s_max
        q_mm = q_mean + q_max

        # mean-max concat
        s_q_concat = torch.cat([s_mm, q_mm], dim = -1)

        # FFN
        lin_res = self.linear(s_q_concat)

        # Softmax
        probs = self.sm(lin_res)
        
        return probs