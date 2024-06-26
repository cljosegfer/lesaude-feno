
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()

        # Define the main branch (left branch)
        self.first_branch = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=16, stride=4, padding=8),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout_rate),
            nn.Conv1d(out_channels, out_channels, kernel_size=1) #Conv 1x1
        )

        # Define the skip connection (right branch)
        self.skip_connection = nn.Sequential(
            nn.MaxPool1d(kernel_size=16, stride=4, padding=8),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

        self.second_branch = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, lower_out, upper_out):
        first_out = self.first_branch(lower_out)
        residual = self.skip_connection(upper_out)
        upper_out = first_out + residual
        lower_out = self.second_branch(upper_out)

        return lower_out, upper_out

class StackedResidual(nn.Module):
    def __init__(self, channels, num_blocks, dropout_rate=0.2):
        super().__init__()

        # Create a list of residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_channels = channels[i]
            out_channels = in_channels + 64
            self.blocks.append(ResidualBlock(in_channels, out_channels, dropout_rate=dropout_rate))


    def forward(self, x):
        lower, upper = x, x
        for i, block in enumerate(self.blocks):
            lower, upper = block(lower, upper)

        return lower

class ResnetJoin(nn.Module):
    def __init__(self, n_classes, num_blocks, channels, dropout_rate=0.2):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=16, stride=4, padding=8, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.residual_blocks = StackedResidual(channels, num_blocks, dropout_rate=dropout_rate)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(960, n_classes)
        self.feno_decoder = nn.Linear(960, 2)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.flatten(x)
        logits = self.linear(x)

        feno = self.feno_decoder(x)

        return {'logits': logits, 'feno': feno}