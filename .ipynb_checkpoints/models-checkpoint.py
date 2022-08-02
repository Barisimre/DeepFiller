from torch import nn

class Classifier(nn.Module):
    # PatchGan 70x70

    def __init__(self):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 300, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(300),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(300, 300, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(300),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(300, 300, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(300),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(300, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):

    def __init__(self, dim):
        super(ResnetBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            # Maybe use dropout here?
            # nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.model(x)


class Generator(nn.Module):
    # Resnet with 9 blocks

    def __init__(self, resnet_blocks=15, features=300):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        )
        self.downSample = nn.Sequential(
            # Down sampling
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, features, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(True),
        )
        self.resnet = nn.Sequential(
            # Resnet blocks
            *[ResnetBlock(features) for _ in range(resnet_blocks)],
        )
        self.upSample = nn.Sequential(
            # Upsamling
            nn.ConvTranspose2d(features, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        )
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downSample(x)
        x = self.resnet(x)
        x = self.upSample(x)
        x = self.final(x)
        return x