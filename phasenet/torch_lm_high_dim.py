# FashionMnist VQ experiment with various settings.
# From https://github.com/minyoungg/vqtorch/blob/main/examples/autoencoder.py

from tqdm.auto import trange
# from typing import Variable
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
import bert_model
from bert_model import BERT
from postprocess import extract_picks
from detect_peaks import detect_peaks
from gaussian import to_gaussian
import numpy


def to_gaussian_batch(one_hot_vectors, std_dev: float = 1.0):
    """
    Convert a batch of one-hot vectors to a batch of Gaussian distribution vectors.

    :param one_hot_vectors: A 2D tensor of shape [batch_size, length] containing one-hot vectors.
    :param std_dev: Standard deviation for the Gaussian distribution.
    :return: A 2D tensor of Gaussian distribution vectors.
    """
    gaussian_batch = []

    for one_hot_vector in one_hot_vectors:
        mean_idx = torch.argmax(one_hot_vector).item()
        indices = torch.arange(one_hot_vector.size(0), dtype=torch.float32)
        gaussian_distribution = torch.exp(-0.5 * ((indices - mean_idx) / std_dev) ** 2)
        gaussian_distribution /= gaussian_distribution.sum()
        gaussian_batch.append(gaussian_distribution)

    return torch.stack(gaussian_batch)

def get_label_tensor(ori_label):
    # get label tensor
    dimension_list = list(ori_label.size())
    Nb = dimension_list[0]
    Nt = dimension_list[1]
    Ns = dimension_list[2]
    Nc = dimension_list[3]
    
    ori_label = ori_label.numpy(force=True)

    phases = ["P", "S"]
    mph={"P": 0.3, "S":0.3}
    mpd = 50
    dt = 0.01
    pre_idx = int(1 / dt)
    post_idx = int(4 / dt)


    picks = []
    for i in range(Nb):
        idxs, probs = detect_peaks(ori_label[i, :, 0, 2], mph=mph["S"], mpd=mpd, show=False) # only S is needed.
        # print(f"idxs: {idxs}, probs : {probs}")
        for l, (phase_index, phase_prob) in enumerate(zip(idxs, probs)):
            phase_index = int(phase_index)
            picks.append(phase_index-3000)
    
    label = torch.tensor(picks).to(device)
    return label

class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, **vq_kwargs):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # changed input channel to 3
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # added padding
                nn.GELU(),
                nn.Conv2d(16, 12, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # added padding
                VectorQuantize(dim=12, accept_image_fmap = True, **vq_kwargs),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1),
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
    # torch.random.manual_seed(seed)

    model = SimpleVQAutoEncoder(codebook_size=num_codes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train(model, train_loader, train_iterations=train_iter)
    torch.save(model, "./model/torch_vqvae_dim12.pt")

    print("Train a BERT")

    model_pred = BERT(hidden = 2256)
    
    
    optimizer = torch.optim.AdamW(model_pred.parameters(), lr=lr)

    model_pred = model_pred.to(device)

    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            yield x.to(device), y.to(device)

    for _ in (pbar := trange(train_iter)):
        optimizer.zero_grad()
        x, y = next(iterate_dataset(train_loader))

        out = torch.nn.functional.pad(x.squeeze(), pad=(0,5), mode="constant", value=0)
        outputs = model_pred(out)

        labels = y[:, :, 0, 2]
        labels = to_gaussian_batch(labels).to(device)
        loss = torch.nn.MSELoss()
        loss_output = loss(outputs, labels)
        loss_output.backward()


        optimizer.step()
        
        ground_truth = labels.max(dim=1)[1].numpy(force=True)
        outputs = outputs.squeeze(1)
        preds = outputs.max(dim=1)[1].numpy(force=True)
        gap = ground_truth - preds
        gap = numpy.absolute(gap)
        accuracy = len(gap[gap<50])/len(gap)

        pbar.set_description(
            f"loss: {loss_output.item():.3f} | "
            f"accuracy: {accuracy:.3f} | "
        )


