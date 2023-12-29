# add project root to path
import sys
import pathlib

root = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
sys.path.append(str(root))

import torch

from src.transformer.model import ScaledDotProductAttention, MultiHeadAttention


def test_scaled_dot_product_attention():
    n_batch = 2
    n_seq = 4
    d_k = 5

    Q = torch.rand(n_batch, n_seq, d_k)
    K = torch.rand(n_batch, n_seq, d_k)
    V = torch.rand(n_batch, n_seq, d_k)

    mask = torch.ones(n_batch, n_seq, n_seq)

    attention = ScaledDotProductAttention()
    output, attention = attention(Q, K, V, mask)

    assert output.size() == (n_batch, n_seq, d_k)
    assert attention.size() == (n_batch, n_seq, n_seq)


def test_multi_head_attention():
    n_batch = 2
    n_seq = 4
    d_model = 64
    num_head = 8

    Q = torch.rand(n_batch, n_seq, d_model)
    K = torch.rand(n_batch, n_seq, d_model)
    V = torch.rand(n_batch, n_seq, d_model)

    attention = MultiHeadAttention(d_model, num_head)
    output = attention(Q, K, V)

    assert output.size() == (n_batch, n_seq, d_model)
