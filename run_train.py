import json

import torch
from thop import profile, clever_format

from configs.train_configs import TrainConfig
from models.fspen import FullSubPathExtension


if __name__ == "__main__":
    configs = TrainConfig()
    with open("config.json", mode="w", encoding="utf-8") as file:
        json.dump(configs.__dict__, file, indent=4)

    model = FullSubPathExtension(configs)

    batch = 1
    groups = configs.dual_path_extension["parameters"]["groups"]
    inter_hidden_size = configs.dual_path_extension["parameters"]["inter_hidden_size"]
    num_modules = configs.dual_path_extension["num_modules"]
    num_bands = sum(configs.bands_num_in_groups)

    in_wav = torch.randn(1, configs.train_points)
    complex_spectrum = torch.stft(in_wav, n_fft=configs.n_fft, hop_length=configs.hop_length,
                                  window=torch.hamming_window(configs.n_fft), return_complex=True)  # (B, F, T)
    amplitude_spectrum = torch.abs(complex_spectrum)

    complex_spectrum = torch.view_as_real(complex_spectrum)  # (B, F, T, 2)
    complex_spectrum = torch.permute(complex_spectrum, dims=(0, 2, 3, 1))
    _, frames, channels, frequency = complex_spectrum.shape
    complex_spectrum = torch.reshape(complex_spectrum, shape=(batch, frames, channels, frequency))
    amplitude_spectrum = torch.permute(amplitude_spectrum, dims=(0, 2, 1))
    amplitude_spectrum = torch.reshape(amplitude_spectrum, shape=(batch, frames, 1, frequency))
    #
    in_hidden_state = [[torch.zeros(1, batch * num_bands, inter_hidden_size//groups) for _ in range(groups)]
                       for _ in range(num_modules)]

    flops, params = profile(model, inputs=(complex_spectrum, amplitude_spectrum, in_hidden_state))
    flops, params = clever_format(nums=[flops, params], format="%0.4f")
    print(f"flops: {flops} \nparams: {params}")
