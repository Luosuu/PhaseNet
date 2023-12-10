import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from seismic_dataset import SeismicDataset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Conv1DModule(nn.Module):
    def __init__(self, input_channels=3, output_features=512):
        super(Conv1DModule, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)

        # Adaptive pooling layer to output a fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer to get the desired output features
        self.fc = nn.Linear(512, output_features)

    def forward(self, x):
        # Assuming x shape is [batch_size, sequence_length, 1, channels]
        # Reshape to [batch_size, channels, sequence_length]
        x = x.squeeze(2).permute(0, 2, 1)

        # Apply convolutional layers with ReLU and optional pooling layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Apply adaptive pooling to get fixed size output
        x = self.adaptive_pool(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer to get the desired output features
        x = self.fc(x)

        return x

class RNNModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, num_layers=1, num_classes=512):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Pass the output of the last time step to the classifier
        out = self.fc(out[:, -1, :])
        return out

class ReverseConv1DModule(nn.Module):
    def __init__(self, input_features=512, output_channels=3, output_length=9001):
        super(ReverseConv1DModule, self).__init__()

        self.fc = nn.Linear(input_features, 256 * 56)  # Start with an upscale

        # Upscaling layers
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=8, stride=4, padding=2)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2)
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2)
        self.deconv4 = nn.ConvTranspose1d(32, output_channels, kernel_size=8, stride=4, padding=2)

        # Calculate the additional upsampling needed after deconv layers
        upscaled_size = 56 * 4 * 4 * 4 * 4  # Based on the strides used
        if upscaled_size < output_length:
            additional_upscale = (output_length // upscaled_size) + 1
            self.final_deconv = nn.ConvTranspose1d(output_channels, output_channels, kernel_size=additional_upscale, stride=additional_upscale - 1, padding=1)
        else:
            self.final_deconv = None

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = x.view(x.size(0), 256, -1)

        # Transposed convolution layers
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))

        if self.final_deconv is not None:
            x = self.final_deconv(x)

        # Ensure the output is the correct size
        current_length = x.size(2)
        if current_length < 9001:
            x = F.pad(x, (0, 9001 - current_length))
        elif current_length > 9001:
            x = x[:, :, :9001]

        x = x.unsqueeze(2)  # Add the singleton dimension
        x = x.permute(0, 3, 2, 1)
        return x


class CombinedModel(nn.Module):
    def __init__(self, conv_module, rnn_module, reverse_conv_module, downsampling):
        super(CombinedModel, self).__init__()
        self.conv_module = conv_module
        self.rnn_module = rnn_module
        self.reverse_conv_module = reverse_conv_module
        self.downsampling = downsampling

    def forward(self, x):
        x = self.downsampling(x)
        # Pass input through Conv1D module
        conv_output = self.conv_module(x)

        # Reshape output for RNN input
        rnn_input = conv_output.unsqueeze(1)  # Add a sequence dimension

        # Pass output to RNN module
        rnn_output = self.rnn_module(rnn_input)

        # Pass RNN output through reverse convolution module
        reverse_conv_output = self.reverse_conv_module(rnn_output)

        return reverse_conv_output


class Downsample(nn.Module):
    def __init__(self, target_length=512):
        super(Downsample, self).__init__()
        self.target_length = target_length

    def forward(self, x):
        # Assuming x is of shape [batch_size, sequence_length, 1, channels]
        # Reshape to [batch_size, channels, sequence_length]
        x = x.squeeze(2).permute(0, 2, 1)

        # Calculate the downsampling ratio
        current_length = x.shape[2]
        downsampling_ratio = current_length / self.target_length

        # Calculate kernel size and stride
        # Using a larger kernel size can help in achieving smoother downsampling
        kernel_size = int(downsampling_ratio)
        stride = kernel_size - 1  # Ensures overlapping pooling to avoid too much reduction

        # If the calculated stride is zero or negative, set it to 1
        stride = max(1, stride)

        # Apply average pooling
        x = F.avg_pool1d(x, kernel_size, stride=stride)

        # Check and adjust the length if it doesn't match the target length exactly
        if x.shape[2] != self.target_length:
            # Perform additional padding or trimming as necessary
            delta = self.target_length - x.shape[2]
            if delta > 0:
                # Padding if the sequence is shorter than desired
                x = F.pad(x, (0, delta))
            else:
                # Trim the sequence if it's longer than desired
                x = x[:, :, :self.target_length]

        # Reshape back to the original format
        x = x.permute(0, 2, 1).unsqueeze(2)  # Shape becomes [batch_size, target_length, 1, channels]

        return x

if __name__ == '__main__':

    ddp_setup()

    data_dir = './dataset/waveform_train/'  
    data_list = './dataset/waveform.csv'
    dataset = SeismicDataset(data_dir = data_dir, data_list = data_list)
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

    conv_module = Conv1DModule()
    rnn_module = RNNModel()
    reverse_conv_module = ReverseConv1DModule()
    downsampling = Downsample()
    model = CombinedModel(conv_module, rnn_module, reverse_conv_module, downsampling)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_function = nn.BCEWithLogitsLoss() 

    gpu_id = int(os.environ["LOCAL_RANK"])
    model = model.to(gpu_id)


    epoch_num = 10
    for epoch in range(epoch_num):
        total_loss = 0
        for idx, batch in enumerate(data_loader):
            optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.float()
            labels = labels.float()
            inputs = inputs.to(gpu_id)
            labels = labels.to(gpu_id)
            # inputs & labels: [batch, time, station, channel]

            outputs = model(inputs)

            # postprocessed_labels = postprocess(labels)
            loss = loss_function(outputs.to(gpu_id), labels)
            loss.backward()
            print(loss)
            # total_loss += loss
            optimizer.step()

        #average_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epoch_num} Completed.")