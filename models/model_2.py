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
        super().__init()
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

class Model2(nn.Module):
    def __init__(self, tbd):
        super().__init__()
        # ENCODE
        # TODO TBD

        # First BiLSTMs
        # TODO TBD

        # First Attention Layer
        self.attention_start = Attention()

        # Second BiLSTMs
        self.lstm2_s = BiLSTM()
        self.lstm2_q = BiLSTM()
        
        # Second Attention Layer
        self.attention_s = Attention()
        self.attention_q = Attention()

        # FFN
        self.linear = nn.Linear()

        self.sm = nn.Softmax(dim = -1)

    def forward(self, s, q):
        # TODO figure out mask situation
        # TODO Mask is 1 if the value doesn't exist, and 0 if it does
        # I.e. if sentence is too short, 1's will be in the place where the sentence does
        # Not exist

        # Encoding

        # First BiLSTM
        lstm_1_s = None
        lstm_1_q = None

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