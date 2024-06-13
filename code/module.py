import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        x  =  self.layer_norm(x)
        return self.fc(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        x  =  self.layer_norm(x)
        return self.fc(x)
    
    
def create_models():
    generator = Generator(input_dim=1009, output_dim=140).to(device)
    discriminator = Discriminator(input_dim=140).to(device)
    return generator, discriminator