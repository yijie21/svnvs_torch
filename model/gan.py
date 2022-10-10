import torch
import torch.nn as nn


# TODO: 完全不知道这里面的kernel_size=4之类的操作是什么意思
class Generator(nn.Module):
    def __init__(self,
                 ngf=64):
        super().__init__()

        self.encoder_layer_specs = [
            ngf * 2,
            ngf * 4,
            ngf * 8,
            ngf * 8,
            ngf * 8
        ]

        self.decoder_layer_specs = [
            (ngf * 8, 0.5),
            (ngf * 8, 0.5),
            (ngf * 4, 0.0),
            (ngf * 2, 0.0),
            (ngf * 1, 0.0)
        ]

        # encoder for x
        # TODO: 卷积核为2的怎么说?
        self.encoder_x_1 = nn.Conv2d(3, ngf, 4, 2, 1)
        self.encoder_x_2 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(ngf, self.encoder_layer_specs[0], 4, 2, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[0])
        )
        self.encoder_x_3 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(self.encoder_layer_specs[0], self.encoder_layer_specs[1], 4, 2, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[1])
        )
        self.encoder_x_4 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(self.encoder_layer_specs[1], self.encoder_layer_specs[2], 4, 2, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[2])
        )
        self.encoder_x_5 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(self.encoder_layer_specs[2], self.encoder_layer_specs[3], 4, 2, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[3])
        )
        self.encoder_x_6 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(self.encoder_layer_specs[3], self.encoder_layer_specs[4], 4, 2, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[4])
        )
        self.encoder_x = nn.ModuleDict({
            'encoder_1': self.encoder_x_1,
            'encoder_2': self.encoder_x_2,
            'encoder_3': self.encoder_x_3,
            'encoder_4': self.encoder_x_4,
            'encoder_5': self.encoder_x_5,
            'encoder_6': self.encoder_x_6
        })
        

        # encoder for y
        self.encoder_y_1 = nn.Conv2d(3, ngf, 4, 2, 1)
        self.encoder_y_2 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(ngf, self.encoder_layer_specs[0], 4, 2, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[0])
        )
        self.encoder_y_3 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(self.encoder_layer_specs[0], self.encoder_layer_specs[1], 4, 2, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[1])
        )
        self.encoder_y_4 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(self.encoder_layer_specs[1], self.encoder_layer_specs[2], 4, 2, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[2])
        )
        self.encoder_y_5 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(self.encoder_layer_specs[2], self.encoder_layer_specs[3], 4, 2, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[3])
        )
        self.encoder_y_6 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            nn.Conv2d(self.encoder_layer_specs[3], self.encoder_layer_specs[4], 4, 2, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[4])
        )
        self.encoder_y = nn.ModuleDict({
            'encoder_1': self.encoder_y_1,
            'encoder_2': self.encoder_y_2,
            'encoder_3': self.encoder_y_3,
            'encoder_4': self.encoder_y_4,
            'encoder_5': self.encoder_y_5,
            'encoder_6': self.encoder_y_6
        })
        

        # confidence generator
        self.confidence_generator_1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2 // 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 2 // 4),
            nn.ReLU(),
            nn.Conv2d(ngf * 2 // 4, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.confidence_generator_2 = nn.Sequential(
            nn.Conv2d(self.encoder_layer_specs[0] * 2, self.encoder_layer_specs[0] * 2 // 4, 3, 1, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[0] * 2 // 4),
            nn.ReLU(),
            nn.Conv2d(self.encoder_layer_specs[0] * 2 // 4, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.confidence_generator_3 = nn.Sequential(
            nn.Conv2d(self.encoder_layer_specs[1] * 2, self.encoder_layer_specs[1] * 2 // 4, 3, 1, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[1] * 2 // 4),
            nn.ReLU(),
            nn.Conv2d(self.encoder_layer_specs[1] * 2 // 4, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.confidence_generator_4 = nn.Sequential(
            nn.Conv2d(self.encoder_layer_specs[2] * 2, self.encoder_layer_specs[2] * 2 // 4, 3, 1, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[2] * 2 // 4),
            nn.ReLU(),
            nn.Conv2d(self.encoder_layer_specs[2] * 2 // 4, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.confidence_generator_5 = nn.Sequential(
            nn.Conv2d(self.encoder_layer_specs[3] * 2, self.encoder_layer_specs[3] * 2 // 4, 3, 1, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[3] * 2 // 4),
            nn.ReLU(),
            nn.Conv2d(self.encoder_layer_specs[3] * 2 // 4, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.confidence_generator_6 = nn.Sequential(
            nn.Conv2d(self.encoder_layer_specs[4] * 2, self.encoder_layer_specs[4] * 2 // 4, 3, 1, 1),
            nn.BatchNorm2d(self.encoder_layer_specs[4] * 2 // 4),
            nn.ReLU(),
            nn.Conv2d(self.encoder_layer_specs[4] * 2 // 4, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.confidence_generator = nn.ModuleDict({
            'confidence_generator_1': self.confidence_generator_1,
            'confidence_generator_2': self.confidence_generator_2,
            'confidence_generator_3': self.confidence_generator_3,
            'confidence_generator_4': self.confidence_generator_4,
            'confidence_generator_5': self.confidence_generator_5,
            'confidence_generator_6': self.confidence_generator_6
        })
        
        
        # decoder
        self.decoder_1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(self.encoder_layer_specs[4], self.decoder_layer_specs[0][0], 4, 2, 1),
            nn.BatchNorm2d(self.decoder_layer_specs[0][0]),
            nn.Dropout2d(self.decoder_layer_specs[0][1]) if self.decoder_layer_specs[0][1] > 0.0 else nn.Identity()
        )
        self.decoder_2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(self.encoder_layer_specs[3] + self.decoder_layer_specs[0][0], self.decoder_layer_specs[1][0], 4, 2, 1),
            nn.BatchNorm2d(self.decoder_layer_specs[1][0]),
            nn.Dropout2d(self.decoder_layer_specs[1][1]) if self.decoder_layer_specs[1][1] > 0.0 else nn.Identity()
        )
        self.decoder_3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(self.encoder_layer_specs[2] + self.decoder_layer_specs[1][0], self.decoder_layer_specs[2][0], 4, 2, 1),
            nn.BatchNorm2d(self.decoder_layer_specs[2][0]),
            nn.Dropout2d(self.decoder_layer_specs[2][1]) if self.decoder_layer_specs[2][1] > 0.0 else nn.Identity()
        )
        self.decoder_4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(self.encoder_layer_specs[1] + self.decoder_layer_specs[2][0], self.decoder_layer_specs[3][0], 4, 2, 1),
            nn.BatchNorm2d(self.decoder_layer_specs[3][0]),
            nn.Dropout2d(self.decoder_layer_specs[3][1]) if self.decoder_layer_specs[3][1] > 0.0 else nn.Identity()
        )
        self.decoder_5 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(self.encoder_layer_specs[0] + self.decoder_layer_specs[3][0], self.decoder_layer_specs[4][0], 4, 2, 1),
            nn.BatchNorm2d(self.decoder_layer_specs[4][0]),
            nn.Dropout2d(self.decoder_layer_specs[4][1]) if self.decoder_layer_specs[4][1] > 0.0 else nn.Identity()
        )
        self.decoder = nn.ModuleDict({
            'decoder_1': self.decoder_1,
            'decoder_2': self.decoder_2,
            'decoder_3': self.decoder_3,
            'decoder_4': self.decoder_4,
            'decoder_5': self.decoder_5
        })

        # decoder to generate image
        self.image_decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1),
            nn.Tanh()
        )

        # decoder to generate confidence
        self.confidence_decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1),
            nn.Sigmoid()
        )


    def forward(self, x_B3HW, y_B3HW):
        # encode x and y
        layers_x = []
        layers_y = []
        n_layers = len(self.encoder_x)
        for i in range(n_layers):
            if i == 0:
                layers_x.append(self.encoder_x['encoder_1'](x_B3HW))
                layers_y.append(self.encoder_y['encoder_1'](y_B3HW))
            else:
                layers_x.append(self.encoder_x['encoder_{}'.format(i + 1)](layers_x[i - 1]))
                layers_y.append(self.encoder_y['encoder_{}'.format(i + 1)](layers_y[i - 1]))
        
        # calculate confidence score
        layers = []
        # TODO: confidence这个变量没用过啊
        confidences = []
        for i, (layer_x, layer_y) in enumerate(zip(layers_x, layers_y)):
            x = torch.cat((layer_x, layer_y), 1)
            x = self.confidence_generator['confidence_generator_{}'.format(i + 1)](x)
            confidences.append(x)
            layers.append(x * layer_x + (1 - x) * layer_y)
        
        # decode
        num_decoder_layers = len(layers)
        for i in range(len(self.decoder)):
            skip_layer_no = num_decoder_layers - i - 1
            if i == 0:
                x = layers[-1]
            else:
                x = torch.cat((layers[-1], layers[skip_layer_no]), 1)
            x = self.decoder['decoder_{}'.format(i + 1)](x)
            layers.append(x)

        # generate image
        out_image_B3HW = self.image_decoder(layers[-1])

        # generate confidence
        out_confidence_B1HW = self.confidence_decoder(layers[-1])

        return out_image_B3HW, out_confidence_B1HW


class Discriminator(nn.Module):
    def __init__(self,
                 ndf=64):
        super().__init__()

        # TODO: original repo uses padded input
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 0),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    
    def forward(self, img_B3HW):
        x = self.layer1(img_B3HW)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x