import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_to_match(x, target):
    
            diff_d = target.size(2)-x.size(2)
            diff_h = target.size(3)-x.size(3)
            diff_w = target.size(4)-x.size(4)
    
    
            return F.pad(x, [
    
                diff_w // 2, diff_w- (diff_w//2),
                diff_h // 2, diff_h - (diff_h // 2),
                diff_d // 2, diff_d - (diff_d // 2)
            ])


class dualConv3D(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(dualConv3D, self).__init__()
        self.conv = nn.Sequential(


            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias =False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),


            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias =False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)


        )

    def forward(self, x):

        return self.conv(x)
    
    
class Unet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, init_features=16):
        super(Unet3D, self).__init__()
        features = init_features  # 16 instead of 32
        
        self.encoder1 = dualConv3D(in_channels, features)
        self.encoder2 = dualConv3D(features, features * 2)
        self.encoder3 = dualConv3D(features * 2, features * 4)
        self.encoder4 = dualConv3D(features * 4, features * 8)
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = dualConv3D((features * 4) + (features * 4), features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = dualConv3D((features * 2) + (features * 2), features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = dualConv3D(features + features, features)
        
        self.out_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        x = self.pool(e1)

        e2 = self.encoder2(x)
        x = self.pool(e2)

        e3 = self.encoder3(x)
        x = self.pool(e3)

        b = self.encoder4(x)

        # --- DECODER PATH ---
        d3 = self.upconv3(b)
        d3 = pad_to_match(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = pad_to_match(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = pad_to_match(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)

        return self.out_conv(d1)
    

if __name__ == "__main__":
    model = Unet3D(in_channels=4, out_channels=3)
    # Simulate a single BraTS volume: (Batch=1, Channels=4, D=155, H=240, W=240)
    dummy_x = torch.randn(1, 4, 155, 240, 240)
    
    out = model(dummy_x)
    print(f"Forward pass successful. Output shape: {out.shape}")