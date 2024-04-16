import torch

from torch import nn, Tensor
from modules.en_decoder import FullBandEncoderBlock, FullBandDecoderBlock
from modules.en_decoder import SubBandEncoderBlock, SubBandDecoderBlock
from modules.sequence_modules import DualPathExtensionRNN
from configs.train_configs import TrainConfig


class FullBandEncoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        last_channels = 0
        self.full_band_encoder = nn.ModuleList()
        for encoder_name, conv_parameter in configs.full_band_encoder.items():
            self.full_band_encoder.append(FullBandEncoderBlock(**conv_parameter))
            last_channels = conv_parameter["out_channels"]

        self.global_features = nn.Conv1d(in_channels=last_channels, out_channels=last_channels, kernel_size=1, stride=1)

    def forward(self, complex_spectrum: Tensor):
        """
        :param complex_spectrum: (batch*frame, channels, frequency)
        :return:
        """
        full_band_encodes = []
        for encoder in self.full_band_encoder:
            complex_spectrum = encoder(complex_spectrum)
            full_band_encodes.append(complex_spectrum)

        global_feature = self.global_features(complex_spectrum)

        return full_band_encodes[::-1], global_feature


class SubBandEncoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()

        self.sub_band_encoders = nn.ModuleList()
        for encoder_name, conv_parameters in configs.sub_band_encoder.items():
            self.sub_band_encoders.append(SubBandEncoderBlock(**conv_parameters["conv"]))

    def forward(self, amplitude_spectrum: Tensor):
        """
        :param amplitude_spectrum: (batch * frames, channels, frequency)
        :return:
        """
        sub_band_encodes = list()
        for encoder in self.sub_band_encoders:
            encode_out = encoder(amplitude_spectrum)
            sub_band_encodes.append(encode_out)

        local_feature = torch.cat(sub_band_encodes, dim=2)  # feature cat

        return sub_band_encodes, local_feature


class FullBandDecoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        self.full_band_decoders = nn.ModuleList()
        for decoder_name, parameters in configs.full_band_decoder.items():
            self.full_band_decoders.append(
                FullBandDecoderBlock(**parameters))

    def forward(self, feature: Tensor, encode_outs: list):
        for decoder, encode_out in zip(self.full_band_decoders, encode_outs):
            feature = decoder(feature, encode_out)

        return feature


class SubBandDecoder(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        start_idx = 0
        self.sub_band_decoders = nn.ModuleList()
        for (decoder_name, parameters), bands in zip(configs.sub_band_decoder.items(), configs.bands_num_in_groups):
            end_idx = start_idx + bands
            self.sub_band_decoders.append(SubBandDecoderBlock(start_idx=start_idx, end_idx=end_idx, **parameters))

    def forward(self, feature: Tensor, sub_encodes: list):
        """
        :param feature: (batch*frames, channels, bands)
        :param sub_encodes: [sub_encode_0, sub_encode_1, ...], each element is (batch*frames, channels, sub_bands)
        :return: (batch*frames, full-frequency)
        """
        sub_decoder_outs = []
        for decoder, sub_encode in zip(self.sub_band_decoders, sub_encodes):
            sub_decoder_out = decoder(feature, sub_encode)
            sub_decoder_outs.append(sub_decoder_out)

        sub_decoder_outs = torch.cat(tensors=sub_decoder_outs, dim=1)  # feature cat

        return sub_decoder_outs


class FullSubPathExtension(nn.Module):
    def __init__(self, configs: TrainConfig):
        super().__init__()
        self.full_band_encoder = FullBandEncoder(configs)
        self.sub_band_encoder = SubBandEncoder(configs)

        merge_split = configs.merge_split
        merge_channels = merge_split["channels"]
        merge_bands = merge_split["bands"]
        compress_rate = merge_split["compress_rate"]

        self.feature_merge_layer = nn.Sequential(
            nn.Linear(in_features=merge_channels, out_features=merge_channels//compress_rate),
            nn.ELU(),
            nn.Conv1d(in_channels=merge_bands, out_channels=merge_bands//compress_rate, kernel_size=1, stride=1)
        )

        self.dual_path_extension_rnn_list = nn.ModuleList()
        for _ in range(configs.dual_path_extension["num_modules"]):
            self.dual_path_extension_rnn_list.append(DualPathExtensionRNN(**configs.dual_path_extension["parameters"]))

        self.feature_split_layer = nn.Sequential(
            nn.Conv1d(in_channels=merge_bands//compress_rate, out_channels=merge_bands, kernel_size=1, stride=1),
            nn.Linear(in_features=merge_channels//compress_rate, out_features=merge_channels),
            nn.ELU()
        )

        self.full_band_decoder = FullBandDecoder(configs)
        self.sub_band_decoder = SubBandDecoder(configs)

        self.mask_padding = nn.ConstantPad2d(padding=(1, 0, 0, 0), value=0.0)

    def forward(self, in_complex_spectrum: Tensor, in_amplitude_spectrum: Tensor, hidden_state: list):
        """
        :param in_amplitude_spectrum: (batch, frames, 1, frequency)
        :param hidden_state:
        :param in_complex_spectrum: (batch, frames, channels, frequency)
        :return:
        """
        batch, frames, channels, frequency = in_complex_spectrum.shape
        complex_spectrum = torch.reshape(in_complex_spectrum, shape=(batch * frames, channels, frequency))
        amplitude_spectrum = torch.reshape(in_amplitude_spectrum, shape=(batch*frames, 1, frequency))

        full_band_encode_outs, global_feature = self.full_band_encoder(complex_spectrum)
        sub_band_encode_outs, local_feature = self.sub_band_encoder(amplitude_spectrum)

        merge_feature = torch.cat(tensors=[global_feature, local_feature], dim=2)  # feature cat
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
        first_dim, channels, frequency = split_feature.shape
        split_feature = torch.reshape(split_feature, shape=(first_dim, channels, -1, 2))

        full_band_mask = self.full_band_decoder(split_feature[..., 0], full_band_encode_outs)
        sub_band_mask = self.sub_band_decoder(split_feature[..., 1], sub_band_encode_outs)

        full_band_mask = torch.reshape(full_band_mask, shape=(batch, frames, 2, -1))
        sub_band_mask = torch.reshape(sub_band_mask, shape=(batch, frames, 1, -1))

        # Zero padding in the DC signal part removes the DC component
        full_band_mask = self.mask_padding(full_band_mask)
        sub_band_mask = self.mask_padding(sub_band_mask)

        full_band_out = in_complex_spectrum * full_band_mask
        sub_band_out = in_amplitude_spectrum * sub_band_mask
        # outputs is (batch, frames, 2, frequency), complex style.

        return full_band_out + sub_band_out, out_hidden_state
