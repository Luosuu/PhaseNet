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
import numpy

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
    model = torch.load("./model/torch_vqvae.pt")

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
        x = x.permute(0, 3, 1, 2)

        out, indices, cmt_loss = model.encode(x)
        out = out.squeeze()
        out = torch.nn.functional.pad(out, pad=(0,5), mode="constant", value=0)
        outputs = model_pred(out)

        # labels = get_label_tensor(y)
        labels = y[:, :, 0, 2]
        # print(labels.max(dim=1)[1])
        # print(f"labels: {labels}")
        # outputs = torch.randint(low=0, high=1285, size=(20,)).to(device).to(float) # random guess, loss ~ 500


        # ground_truth = labels.max(dim=1)[1].to(float)
        # preds = outputs.max(dim=1)[1].to(float)

        ground_truth = torch.nn.functional.softmax(labels, dim=1)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        loss_output = (ground_truth - outputs).abs().mean()
        # loss_output = loss(outputs, ground_truth)
        loss_output.backward()

        # loss = (outputs - labels).abs().mean()
        # loss.backward()

        optimizer.step()
        
        ground_truth = labels.max(dim=1)[1].numpy(force=True)
        preds = outputs.max(dim=1)[1].numpy(force=True)
        gap = ground_truth - preds
        gap = numpy.absolute(gap)
        # print(f"gap: {gap}")
        accuracy = len(gap[gap<50])/len(gap)

        print( 
            f"ground_truth: {ground_truth} | \n "
            f"preds: {preds} | \n"
            f"gap: {gap} | \n"
            f"accuracy: {accuracy} | \n"
        )

        pbar.set_description(
            f"loss: {loss_output.item():.3f} | "
        )


