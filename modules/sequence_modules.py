# !/user/bin/env python
# -*-coding:utf-8 -*-

"""
# File : sequence_modules.py
# Time : 2024/4/10 上午9:35
# Author : wukeyi
# version : python3.9
"""
from typing import List

import torch
from torch import nn, Tensor


class GroupRNN(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 groups: int,
                 rnn_type: str,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 batch_first: bool = True):
        super().__init__()
        assert input_size % groups == 0, \
            f"input_size % groups must be equal to 0, but got {input_size} % {groups} = {input_size % groups}"

        self.groups = groups
        self.rnn_list = nn.ModuleList()
        for _ in range(groups):
            self.rnn_list.append(
                getattr(nn, rnn_type)(input_size=input_size // groups, hidden_size=hidden_size//groups,
                                      num_layers=num_layers,
                                      bidirectional=bidirectional, batch_first=batch_first)
            )

    def forward(self, inputs: Tensor, hidden_state: List[Tensor]):
        """
        :param hidden_state: List[state1, state2, ...], len(hidden_state) = groups
        state shape = (num_layers*bidirectional, batch*[], hidden_size) if rnn_type is GRU or RNN, otherwise,
        state = (h0, c0), h0/c0 shape = (num_layers*bidirectional, batch*[], hidden_size).
        :param inputs: (batch, steps, input_size)
        :return:
        """
        outputs = []
        out_states = []
        batch, steps, _ = inputs.shape

        inputs = torch.reshape(inputs, shape=(batch, steps, self.groups, -1))  # (batch, steps, groups, width)
        for idx, rnn in enumerate(self.rnn_list):
            out, state = rnn(inputs[:, :, idx, :], hidden_state[idx])
            outputs.append(out)  # (batch, steps, hidden_size)
            out_states.append(state)  # (num_layers*bidirectional, batch*[], hidden_size)

        outputs = torch.cat(outputs, dim=2)  # (batch, steps, hidden_size * groups)

        return outputs, out_states


class DualPathExtensionRNN(nn.Module):
    def __init__(self, input_size: int,
                 intra_hidden_size: int,
                 inter_hidden_size: int,
                 groups: int,
                 rnn_type: str):
        super().__init__()
        assert rnn_type in ["RNN", "GRU", "LSTM"], f"rnn_type should be RNN/GRU/LSTM, but got {rnn_type}!"

        self.intra_chunk_rnn = getattr(nn, rnn_type)(input_size=input_size, hidden_size=intra_hidden_size,
                                                     num_layers=1, bidirectional=True, batch_first=True)
        self.intra_chunk_fc = nn.Linear(in_features=intra_hidden_size*2, out_features=input_size)
        self.intra_chunk_norm = nn.LayerNorm(normalized_shape=input_size, elementwise_affine=True)

        self.inter_chunk_rnn = GroupRNN(input_size=input_size, hidden_size=inter_hidden_size, groups=groups,
                                        rnn_type=rnn_type)
        self.inter_chunk_fc = nn.Linear(in_features=inter_hidden_size, out_features=input_size)

    def forward(self, inputs: Tensor, hidden_state: List[Tensor]):
        """
        :param hidden_state: List[state1, state2, ...], len(hidden_state) = groups
        state shape = (num_layers*bidirectional, batch*[], hidden_size) if rnn_type is GRU or RNN, otherwise,
        state = (h0, c0), h0/c0 shape = (num_layers*bidirectional, batch*[], hidden_size).
        :param inputs: (B, F, T, N)
        :return:
        """
        B, F, T, N = inputs.shape
        intra_out = torch.transpose(inputs, dim0=1, dim1=2).contiguous()  # (B, T, F, N)
        intra_out = torch.reshape(intra_out, shape=(B * T, F, N))
        intra_out, _ = self.intra_chunk_rnn(intra_out)
        intra_out = self.intra_chunk_fc(intra_out)  # (B, T, F, N)
        intra_out = torch.reshape(intra_out, shape=(B, T, F, N))
        intra_out = torch.transpose(intra_out, dim0=1, dim1=2).contiguous()  # (B, F, T, N)
        intra_out = self.intra_chunk_norm(intra_out)  # (B, F, T, N)

        intra_out = inputs + intra_out  # residual add

        inter_out = torch.reshape(intra_out, shape=(B * F, T, N))  # (B*F, T, N)
        inter_out, hidden_state = self.inter_chunk_rnn(inter_out, hidden_state)
        inter_out = torch.reshape(inter_out, shape=(B, F, T, -1))  # (B, F, T, groups * N)
        inter_out = self.inter_chunk_fc(inter_out)  # (B, F, T, N)

        inter_out = inter_out + intra_out  # residual add

        return inter_out, hidden_state


if __name__ == "__main__":
    test_model = DualPathExtensionRNN(input_size=32, intra_hidden_size=16, inter_hidden_size=16,
                                      groups=8, rnn_type="LSTM")
    test_data = torch.randn(5, 32, 10, 32)
    test_state = [(torch.randn(1, 5*32, 16), torch.randn(1, 5*32, 16)) for _ in range(8)]
    test_out = test_model(test_data, test_state)
