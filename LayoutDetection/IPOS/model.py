import torch
import torch.nn as nn

class IposNet(nn.Module):
    def __init__(self):
        super(IposNet, self).__init__()
        
        # As per paper (pg. 3, eq. 2), leaky ReLU with negative_slope=0.1
        leaky_relu_slope = 0.1
        
        # Convolutional layers from Figure 3
        # Block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(leaky_relu_slope),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192 , kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 128 , kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(128, 256 , kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 256 , kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 512 , kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leaky_relu_slope),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # Block 4
        self.layer4 = nn.Sequential(
            # 4 repetitions of (1x1 conv -> 3x3 conv)
            nn.Conv2d(512, 256 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 512 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 256 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 512 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 256 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 512 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 256 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(256, 512 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            # End of 4 repetitions
            nn.Conv2d(512, 512 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 5
        self.layer5 = nn.Sequential(
            # 2 repetitions of (1x1 conv -> 3x3 conv)
            nn.Conv2d(1024, 512 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(1024, 512 , kernel_size=1, stride=1, padding=0), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(512, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            # End of 2 repetitions
            nn.Conv2d(1024, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(1024, 1024 , kernel_size=3, stride=2, padding=1), nn.LeakyReLU(leaky_relu_slope)
        )
        
        # Block 6
        self.layer6 = nn.Sequential(
            nn.Conv2d(1024, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope),
            nn.Conv2d(1024, 1024 , kernel_size=3, stride=1, padding=1), nn.LeakyReLU(leaky_relu_slope)
        )

        # Fully Connected Layers
        # Output tensor size: S x S x (B*5 + C)
        # For PASCAL VOC: S=7, B=2, C=20. So, 7*7*(2*5 + 20) = 7*7*30 = 1470
        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.drop_out = nn.Dropout(0.5) # Dropout rate 0.5 as per paper
        self.fc2 = nn.Linear(4096, 7*7*30) # Output is linear

    def forward(self, x):
        # print(f"Initial x shape: {x.shape}")
        out = self.layer1(x)
        # print(f"After layer1: {out.shape}")
        out = self.layer2(out)
        # print(f"After layer2: {out.shape}")
        out = self.layer3(out)
        # print(f"After layer3: {out.shape}")
        out = self.layer4(out)
        # print(f"After layer4: {out.shape}")
        out = self.layer5(out)
        # print(f"After layer5: {out.shape}")
        out = self.layer6(out)
        # print(f"After layer6: {out.shape}")
        
        out = torch.flatten(out, start_dim=1) # Flatten while keeping batch dimension
        # print(f"After flatten: {out.shape}")
        
        out = self.fc1(out)
        # print(f"After fc1: {out.shape}")
        out = nn.LeakyReLU(0.1)(out) # Leaky ReLU after first FC
        out = self.drop_out(out)      # Dropout after first FC's activation
        
        out = self.fc2(out)           # Final FC layer (linear activation)
        # print(f"After fc2: {out.shape}")
        
        # Reshape to (N, S, S, (B*5+C)) which is (N, 7, 7, 30) for VOC
        out = out.reshape(-1, 7, 7, 30) 
        # print(f"Final reshaped out: {out.shape}")
        return out


model = IposNet()

dummy_input = torch.randn(1, 3, 448, 448) 

try:
    output = model(dummy_input)
    print(f"Successfully processed input. Output shape: {output.shape}") # Expected: [1, 7, 7, 30]
except Exception as e:
    print(f"Error during model forward pass: {e}")


# Calculate total and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")          # Expected: Around 45-60 million (original YOLO is large)
print(f"Trainable parameters: {trainable_params}")  # Should be same as total_params
