import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self) -> None:
        super(Encoder, self).__init__()
        self.enc1 = nn.Conv2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        return x


class Decoder(nn.Module):
    def __init__(self) -> None:
        super(Decoder, self).__init__()
        self.dec1 = nn.ConvTranspose2d(
            in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, output_padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=8, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return x


class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x
