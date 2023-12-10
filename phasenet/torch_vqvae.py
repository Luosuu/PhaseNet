# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    BitsAndBytesConfig,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from vector_quantize_pytorch import VectorQuantize
from seismic_dataset import SeismicDataset
from rinas.hf.mlm_modules import *
from accelerate import Accelerator
from huggingface_hub import login

class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, **vq_kwargs):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # changed input channel to 3
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # added padding
                nn.GELU(),
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # added padding
                VectorQuantize(dim=1, accept_image_fmap = True, **vq_kwargs),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            ]
        )
        
        return

    def forward(self, x):
        x, indices, commit_loss = self.encode(x)
        x = self.decode(x)

        return x, indices, commit_loss

    def encode(self, x):
        # print(x)
        for layer in self.encoder:
            if isinstance(layer, VectorQuantize):
                x, indices, commit_loss = layer(x)
            else:
                x = layer(x)
        
        return x, indices, commit_loss
    
    def decode(self, x):
        for layer in self.decoder:
            x = layer(x)
        # print(x)
        return x.clamp(-1, 1)


def train(model, train_loader, train_iterations=1000, alpha=10):
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    for _ in (pbar := trange(train_iterations)):
        opt.zero_grad()
        x, y = next(iterate_dataset(train_loader))
        # print(x.shape) torch.Size([154, 3, 9001, 1])
        # print(y.shape)
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        out, indices, cmt_loss = model(x)
        # print(out.shape)
        out = out[:, :, :x.size(2)] # added to match out and x
        rec_loss = (out - x).abs().mean()
        # rec_loss = (out - y).abs().mean()
        (rec_loss + alpha * cmt_loss).backward()

        opt.step()
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"cmt loss: {cmt_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )
    return

if __name__ == '__main__':

    
    lr = 3e-4
    train_iter = 1000
    num_codes = 256
    seed = 1234
    batch_size = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_dir = './dataset/waveform_train/'  
    data_list = './dataset/waveform.csv'


    dataset = SeismicDataset(data_dir = data_dir, data_list = data_list, transform=None)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    

    print("training VQ-VAE")
    torch.random.manual_seed(seed)
    model = SimpleVQAutoEncoder(codebook_size=num_codes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train(model, train_loader, train_iterations=train_iter)