from torch import nn


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv1d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
        #     nn.ReLU(True),
        #     nn.MaxPool1d(2, stride=2),  # b, 16, 5, 5
        #     nn.Conv1d(16, 1, 3, stride=2, padding=1),  # b, 8, 3, 3
        #     nn.ReLU(True),
        #     nn.MaxPool1d(2, stride=1)  # b, 8, 2, 2
        # )
        self.encoder = nn.Sequential(
            # nn.Conv3d(1, 4, 2, stride=1),
            # nn.ReLU(True),
            # nn.Conv3d(4, 8, 2, stride=1),
            # nn.ReLU(True),
            # nn.Conv3d(8, 1, 2, stride=1)

            nn.Conv3d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),  # b, 16, 5, 5
            nn.Conv3d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=1)  # b, 8, 2, 2
        )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
        #     #nn.Tanh()
        # )

    def forward(self, x):
        x = self.encoder(x)
        #x = self.decoder(x)
        return x

# from torch import nn
#
#
# class encDec(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
#             nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
#             nn.ReLU(True),
#             nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
#             nn.Tanh()
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

