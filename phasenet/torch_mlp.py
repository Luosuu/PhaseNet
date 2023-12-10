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
from torch_vqvae import SimpleVQAutoEncoder, train
from postprocess import extract_picks
from detect_peaks import detect_peaks

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

class model_pred(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp_0 = torch.nn.Linear(in_features=2251, out_features=1024)
        self.relu = torch.nn.ReLU()
        self.mlp_1 = torch.nn.Linear(in_features=1024, out_features=1)

    def forward(self, x):
        x = self.mlp_0(x)
        x = self.relu(x)
        x = self.mlp_1(x)
        x = self.relu(x)

        return x


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
    # model = SimpleVQAutoEncoder(codebook_size=num_codes).to(device)
    # opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # train(model, train_loader, opt, device, train_iterations=train_iter)
    # torch.save(obj=model, f="./model/torch_vqvae.pt")
    model = torch.load("./model/torch_vqvae.pt")

    print("Train a MLP basline")

    model_pred = model_pred()
    
    
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
        accum = 0
        for step in range(100):
            optimizer.zero_grad()
            x, y = next(iterate_dataset(train_loader))
            x = x.permute(0, 3, 1, 2)

            out, indices, cmt_loss = model.encode(x)
            out = out.squeeze()
            outputs = model_pred(out)

            labels = get_label_tensor(y)
            outputs = torch.randint(low=0, high=1285, size=(20,)).to(device).to(float) # random guess, loss ~ 500
            loss = (outputs - labels).abs().mean()
            # loss.backward()
            # optimizer.step()
            accum+=loss.item()

        pbar.set_description(
                f"aver_loss: {accum/100:.3f} | "
            )