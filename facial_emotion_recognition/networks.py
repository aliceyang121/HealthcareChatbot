import torch.nn as nn


class NetworkV2(nn.Module):
    def __init__(self, in_c, nl, out_f):
        super(NetworkV2, self).__init__()
        self.in_c, self.nl, self.out_f = in_c, nl, out_f

        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=self.in_c, out_channels=self.nl, kernel_size=(3, 3), stride=1, padding=0),
            nn.BatchNorm2d(self.nl),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=self.nl, out_channels=2*self.nl, kernel_size=(3, 3), stride=2, padding=0),
            nn.BatchNorm2d(2*self.nl),
            nn.ReLU(inplace=True)
        )

        self.conv_3 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=2*self.nl, out_channels=4 * self.nl, kernel_size=(3, 3), stride=2, padding=0),
            nn.BatchNorm2d(4 * self.nl),
            nn.ReLU(inplace=True)
        )

        self.conv_4 = nn.Sequential(
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(in_channels=4 * self.nl, out_channels=8 * self.nl, kernel_size=(3, 3), stride=2, padding=0),
            nn.BatchNorm2d(8 * self.nl),
            nn.ReLU(inplace=True)
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=256*6*6, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=self.out_f),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = x.reshape((x.shape[0], -1))
        x = self.linear(x)
        return x