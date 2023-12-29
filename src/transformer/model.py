# implementing 'Attention is all you need' paper
# https://arxiv.org/abs/1706.03762
# reference: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

from torch import nn
import torch


# CONFUSION ALERT:
# this doesn't have weights. it is just a function.
# same as `torch.nn.functional.scaled_dot_product_attention`
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, seq_len, d_k or d_v)
        # self.mask: (seq_len, seq_len)

        self._check_dimensions(Q, K, V, mask)

        d_k = Q.size(-1)
        QK = torch.matmul(Q, K.transpose(-2, -1))
        scaled_QK = QK / (d_k**0.5)

        if mask is not None:
            # set to -inf where mask is 0 to make softmax output 0
            scaled_QK = scaled_QK.masked_fill(mask == 0, -float("inf"))

        attention = torch.softmax(scaled_QK, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, V)

        return output, attention

    def _check_dimensions(self, Q, K, V, mask):
        assert Q.dim() == 3
        assert K.dim() == 3
        assert V.dim() == 3
        assert Q.size() == K.size() == V.size()
        assert Q.size(0) == K.size(0) == V.size(0)
        assert Q.size(-1) == K.size(-1)
        assert mask is None or mask.dim() == 3
        assert mask is None or mask.size(0) == Q.size(0)
        assert mask is None or mask.size(1) == Q.size(-2)
        assert mask is None or mask.size(2) == Q.size(-2)


# CONFUSION ALERT:
# it is different from the uvadlc implementation.
# uvadlc implementation stacks all weights for multihead Linear layers.
# however, this implementation will have connections between each head, which makes less sense to me.
# So, I use nn.ModuleList to have separate weights for each head.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head, dropout_p=0.2):
        super().__init__()
        assert d_model % num_head == 0
        self.d_k = d_model // num_head
        self.d_v = d_model // num_head
        self.num_heads = num_head
        self.d_model = d_model

        self.dropout_p = dropout_p

        self.W_Qs = self._generate_linear_layers(d_model, self.d_k, num_head)
        self.W_Ks = self._generate_linear_layers(d_model, self.d_k, num_head)
        self.W_Vs = self._generate_linear_layers(d_model, self.d_v, num_head)
        self.W_O = nn.Linear(num_head * self.d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout_p)

    def _generate_linear_layers(self, d_input, d_output, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(d_input, d_output, bias=False))
        return nn.ModuleList(layers)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, seq_len, d_model)
        # self.mask: (batch_size, seq_len, seq_len)

        outputs = []

        for i in range(self.num_heads):
            _Qi = self.W_Qs[i](Q)
            _Ki = self.W_Ks[i](K)
            _Vi = self.W_Vs[i](V)

            _output, _ = self.attention(_Qi, _Ki, _Vi, mask)
            outputs.append(_output)

        output = torch.cat(outputs, dim=-1)
        output = self.W_O(output)  # (batch_size, seq_len, d_model)

        return output
