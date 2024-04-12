import torch

from torch import nn, Tensor
from modules.en_decoder import FullBandEncoderBlock, FullBandDecoderBlock
from modules.en_decoder import SubBandEncoderBlock, SubBandDecoderBlock
from modules.sequence_modules import DualPathExtensionRNN


class FullBandEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.full_band_encoder1 = FullBandEncoderBlock(in_channels=in_channels, out_channels=in_channels * 2,
                                                       kernel_size=6, stride=2)
        self.full_band_encoder2 = FullBandEncoderBlock(in_channels=in_channels * 2, out_channels=in_channels * 8,
                                                       kernel_size=8, stride=4)
        self.full_band_encoder3 = FullBandEncoderBlock(in_channels=in_channels * 8, out_channels=in_channels * 16,
                                                       kernel_size=6, stride=2)
        self.global_features = nn.Conv1d(in_channels=in_channels * 16, out_channels=in_channels * 16,
                                         kernel_size=1, stride=1)

    def forward(self, complex_spectrum: Tensor):
        """
        :param complex_spectrum: (batch*frame, channels, frequency)
        :return:
        """
        encode_out1 = self.full_band_encoder1(complex_spectrum)
        encode_out2 = self.full_band_encoder2(encode_out1)
        encode_out3 = self.full_band_encoder3(encode_out2)
        global_feature = self.global_features(encode_out3)

        return encode_out1, encode_out2, encode_out3, global_feature


class SubBandEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.sub_band_encoder1 = SubBandEncoderBlock(start_frequency=0, end_frequency=15, in_channels=in_channels,
                                                     out_channels=32, kernel=4, stride=2)
        self.sub_band_encoder2 = SubBandEncoderBlock(start_frequency=15, end_frequency=31, in_channels=in_channels,
                                                     out_channels=32, kernel=7, stride=3)
        self.sub_band_encoder3 = SubBandEncoderBlock(start_frequency=31, end_frequency=63, in_channels=in_channels,
                                                     out_channels=32, kernel=11, stride=5)
        self.sub_band_encoder4 = SubBandEncoderBlock(start_frequency=63, end_frequency=127, in_channels=in_channels,
                                                     out_channels=32, kernel=20, stride=10)
        self.sub_band_encoder5 = SubBandEncoderBlock(start_frequency=127, end_frequency=256, in_channels=in_channels,
                                                     out_channels=32, kernel=40, stride=20)

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch * frames, channels, frequency)
        :return:
        """
        encode_out1 = self.sub_band_encoder1(amplitude_spectrum)
        encode_out2 = self.sub_band_encoder2(amplitude_spectrum)
        encode_out3 = self.sub_band_encoder3(amplitude_spectrum)
        encode_out4 = self.sub_band_encoder4(amplitude_spectrum)  # (batch*frames, out_channels, bands)
        encode_out5 = self.sub_band_encoder5(amplitude_spectrum)

        local_feature = torch.cat([encode_out1, encode_out2, encode_out3, encode_out4, encode_out5], dim=2)

        return encode_out1, encode_out2, encode_out3, encode_out4, encode_out5, local_feature


class FullBandDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.full_band_decoder1 = FullBandDecoderBlock(in_channels=64, out_channels=16, kernel_size=6, stride=2)
        self.full_band_decoder2 = FullBandDecoderBlock(in_channels=32, out_channels=4, kernel_size=8, stride=4)
        self.full_band_decoder3 = FullBandDecoderBlock(in_channels=8, out_channels=2, kernel_size=6, stride=2)

    def forward(self, split_feature: Tensor, encode_out1: Tensor, encode_out2: Tensor, encode_out3: Tensor):
        feature = torch.cat([split_feature, encode_out3], dim=1)
        feature = self.full_band_decoder1(feature)

        feature = torch.cat([feature, encode_out2], dim=1)
        feature = self.full_band_decoder2(feature)

        feature = torch.cat([feature, encode_out1], dim=1)
        feature = self.full_band_decoder3(feature)

        return feature


class SubBandDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_band_decoder1 = SubBandDecoderBlock(in_channels=64, out_channels=2)
        self.sub_band_decoder2 = SubBandDecoderBlock(in_channels=64, out_channels=3)
        self.sub_band_decoder3 = SubBandDecoderBlock(in_channels=64, out_channels=5)
        self.sub_band_decoder4 = SubBandDecoderBlock(in_channels=64, out_channels=10)
        self.sub_band_decoder5 = SubBandDecoderBlock(in_channels=64, out_channels=20)

    def forward(self, ):


class FullSubPathExtension(nn.Module):
    def __init__(self, in_channels: int, groups: int, rnn_type: str, num_rnn_modules: int):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(in_channels=in_channels)
        self.sub_band_encoder = SubBandEncoder(in_channels=in_channels)
        self.feature_merge_layer = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ELU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, stride=1)
        )

        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(num_rnn_modules):
            self.dual_path_extension_rnn_list.append(
                DualPathExtensionRNN(input_size=32, intra_hidden_size=16,
                                     inter_hidden_size=16,
                                     groups=groups, rnn_type=rnn_type)
            )

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
            nn.Linear(in_features=32, out_features=64),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder()

    def forward(self, complex_spectrum: Tensor, hidden_state: list):
        """
        :param hidden_state:
        :param complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = complex_spectrum.shape
        complex_spectrum = torch.reshape(complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.abs(complex_spectrum)

        *full_band_encode_out, global_feature = self.full_band_encoder(complex_spectrum)
        *sub_band_encode_out, local_feature = self.sub_band_encoder(amplitude_spectrum)

        merge_feature = torch.cat([global_feature, local_feature], dim=2)  # frequency cat
        merge_feature = self.feature_merge_layer(merge_feature)
        # (batch*frames, channels, frequency) -> (batch*frames, channels//2, frequency//2)
        _, channels, frequency = merge_feature.shape
        merge_feature = torch.reshape(merge_feature, shape=(batch, frames, channels, frequency))
        merge_feature = torch.permute(merge_feature, dims=(0, 3, 1, 2)).contiguous()
        # (batch, frequency, frames, channels)
        out_hidden_state = list()
        for idx, rnn_layer in enumerate(self.dual_path_extension_rnn_list):
            merge_feature, state = rnn_layer(merge_feature, hidden_state[idx])
            out_hidden_state.append(state)

        merge_feature = torch.permute(merge_feature, dims=(0, 2, 3, 1)).contiguous()
        merge_feature = torch.reshape(merge_feature, shape=(batch * frames, channels, frequency))

        split_feature = self.feature_split_layer(merge_feature)
        split_feature = torch.reshape(split_feature, shape=(batch * frames, channels, -1, 2))

        full_band_decoder_out = self.full_band_decoder(split_feature[..., 0], *full_band_encode_out)
