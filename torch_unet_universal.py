import torch
from torchvision.transforms.functional import center_crop
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 batch_norm,
                 kernel_size=3,
                 stride=1,
                 padding='same'):
        super().__init__()  # Это эквивалентно super(CNNBlock, self).__init__()

        if batch_norm:
            self.seq_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,  kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.seq_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.seq_block(x)
        return x


class CNNBlocks(nn.Module):
    def __init__(self,
                 n_conv,
                 in_channels,
                 out_channels,
                 batch_norm,
                 padding,
                 kernel_size=3):
        super(CNNBlocks, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_conv):

            self.layers.append(CNNBlock(in_channels, out_channels, batch_norm, padding=padding, kernel_size=kernel_size))
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 batch_norm,
                 drop_out,
                 padding,
                 kernel_size=3,
                 downhill=4):
        super(Encoder, self).__init__()
        self.enc_layers = nn.ModuleList()

        for _ in range(downhill):
            if drop_out:
                self.enc_layers += [
                        CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, batch_norm=batch_norm, padding=padding, kernel_size=kernel_size),
                        nn.MaxPool2d(2, 2),
                        nn.Dropout(p=0.5)
                    ]
            else:
                self.enc_layers += [
                    CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, batch_norm=batch_norm,
                              padding=padding, kernel_size=kernel_size),
                    nn.MaxPool2d(2, 2)
                ]

            in_channels = out_channels
            out_channels *= 2
        # doubling the dept of the last CNN block
        self.enc_layers.append(CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, batch_norm=batch_norm, padding=padding))

    def forward(self, x):
        route_connection = []
        for layer in self.enc_layers:
            if isinstance(layer, CNNBlocks):
                x = layer(x)
                route_connection.append(x)
            else:
                x = layer(x)
        return x, route_connection


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 exit_channels,
                 batch_norm,
                 drop_out,
                 padding,
                 uphill=4):
        super(Decoder, self).__init__()
        self.exit_channels = exit_channels
        self.layers = nn.ModuleList()

        for i in range(uphill):
            if drop_out:
                self.layers += [
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    nn.Dropout(p=0.5),
                    CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, batch_norm=batch_norm, padding=padding),
                ]
            else:
                self.layers += [
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                    CNNBlocks(n_conv=2, in_channels=in_channels, out_channels=out_channels, batch_norm=batch_norm,
                              padding=padding),
                ]
            in_channels //= 2
            out_channels //= 2

        # cannot be a CNNBlock because it has ReLU incorpored
        # cannot append nn.Sigmoid here because you should be later using
        # BCELoss () which will trigger the amp error "are unsafe to autocast".
        self.layers.append(
            nn.Conv2d(in_channels, exit_channels, kernel_size=1, padding=padding),
        )

    def forward(self, x, routes_connection):
        # pop the last element of the list since
        # it's not used for concatenation
        routes_connection.pop(-1)
        for layer in self.layers:
            if isinstance(layer, CNNBlocks):
                # center_cropping the route tensor to make width and height match
                routes_connection[-1] = center_crop(routes_connection[-1], x.shape[2])
                # concatenating tensors channel-wise
                x = torch.cat([x, routes_connection.pop(-1)], dim=1)
                x = layer(x)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self,
                 in_channels,
                 first_out_channels,
                 exit_channels,
                 batch_norm,
                 drop_out,
                 downhill,
                 kernel_size=3,
                 padding='same',
                 ):
        super(UNET, self).__init__()
        self.encoder = Encoder(in_channels, first_out_channels, batch_norm=batch_norm, drop_out=drop_out, padding=padding, downhill=downhill, kernel_size=kernel_size)
        self.decoder = Decoder(first_out_channels*(2**downhill), first_out_channels*(2**(downhill-1)),
                               exit_channels, batch_norm=batch_norm, drop_out=drop_out, padding=padding, uphill=downhill)
        self.last_act = nn.Sigmoid()

    def forward(self, x):
        enc_out, routes = self.encoder(x)
        out = self.decoder(enc_out, routes)
        out = self.last_act(out)
        return out


if __name__ == '__main__':
    model = UNET(3, 64, 1, True, False, 4, 3)
    print(sum(p.numel() for p in model.parameters()))

