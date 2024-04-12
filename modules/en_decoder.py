# 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

import torch
from torch import nn, Tensor


class FullBandEncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

        self.norm = nn.BatchNorm1d(num_features=out_channels)

        self.activate = nn.ELU()

    def forward(self, complex_spectrum: Tensor):
        """
        :param complex_spectrum: (batch * frames, channels, frequency)
        :return:
        """
        complex_spectrum = self.conv(complex_spectrum)
        complex_spectrum = self.norm(complex_spectrum)
        complex_spectrum = self.activate(complex_spectrum)

        return complex_spectrum


class FullBandDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 2,
                              kernel_size=1, stride=1, padding=padding)
        self.convT = nn.ConvTranspose1d(in_channels // 2, out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding)

        self.norm = nn.BatchNorm1d(num_features=out_channels)
        self.activate = nn.ELU()

    def forward(self, encode_complex_spectrum: Tensor, decode_complex_spectrum):
        """
        :param decode_complex_spectrum: (batch * frames, channels1, frequency)
        :param encode_complex_spectrum: (batch * frames, channels2, frequency)
        :return:
        """
        complex_spectrum = torch.cat([encode_complex_spectrum, decode_complex_spectrum], dim=1)
        complex_spectrum = self.conv(complex_spectrum)
        complex_spectrum = self.convT(complex_spectrum)
        complex_spectrum = self.norm(complex_spectrum)
        complex_spectrum = self.activate(complex_spectrum)

        return complex_spectrum


class SubBandEncoderBlock(nn.Module):
    def __init__(self, start_frequency: int,
                 end_frequency: int,
                 in_channels: int,
                 out_channels: int,
                 kernel: int,
                 stride: int):
        super().__init__()
        self.start_frequency = start_frequency
        self.end_frequency = end_frequency

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                              padding=(kernel - 1) // 2)
        self.activate = nn.ReLU()

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch*frames, channels, frequency)
        :return:
        """
        sub_spectrum = amplitude_spectrum[:, :, self.start_frequency:self.end_frequency]

        sub_spectrum = self.conv(sub_spectrum)  # (batch*frames, out_channels, sub_bands)
        sub_spectrum = self.activate(sub_spectrum)

        return sub_spectrum


class SubBandDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.activate = nn.ReLU()

    def forward(self, encode_amplitude_spectrum: Tensor, decode_amplitude_spectrum: Tensor):
        """

        :param encode_amplitude_spectrum: (batch * frames, channels, bands)
        :param decode_amplitude_spectrum:
        :return:
        """
        spectrum = torch.cat([encode_amplitude_spectrum, decode_amplitude_spectrum], dim=1)
        spectrum = torch.transpose(spectrum, dim0=1, dim1=2).contiguous()   # (*, bands, channels)

        spectrum = self.fc(spectrum)  # (*, bands, band-width)
        spectrum = self.activate(spectrum)

        return spectrum
