from monai.networks.nets import UNet
import torch.nn as nn
import torch

class MyUNet(nn.Module):
    def __init__(self):
        super(MyUNet, self).__init__()
        self.unet = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=256,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,  # Number of residual units
            dropout=0.1
        )

    def forward(self, x):
        return self.unet(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyUNet()
    input_tensor = torch.rand((30, 18, 128, 128)) # Example input
    encoder_output = model(input_tensor)
    print(encoder_output.shape)  # [30, 256, 128, 128]
