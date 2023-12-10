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

    y_max = 0

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
        x, y = next(iterate_dataset(train_loader))
        x = x.permute(0, 3, 1, 2)

        # y = y.permute(0, 3, 1, 2)

        # y = y.squeeze()
        # print(y)
        # print(y.shape)

        dimension_list = list(y.size())
        Nb = dimension_list[0]
        Nt = dimension_list[1]
        Ns = dimension_list[2]
        Nc = dimension_list[3]
        
        y = y.numpy(force=True)

        
        # print(f"Nb: {Nb}, Nt: {Nt}, Ns: {Ns}, Nc: {Nc}")
        phases = ["P", "S"]
        mph={"P": 0.3, "S":0.3}
        mpd = 50
        dt = 0.01
        pre_idx = int(1 / dt)
        post_idx = int(4 / dt)


        picks = []
        for i in range(Nb):
            idxs, probs = detect_peaks(y[i, :, 0, 2], mph=mph["S"], mpd=mpd, show=False) # only S is needed.
            # print(f"idxs: {idxs}, probs : {probs}")
            for l, (phase_index, phase_prob) in enumerate(zip(idxs, probs)):
                if phase_index > y_max:
                    y_max = phase_index
                phase_index = int(phase_index)
                picks.append(phase_index-3000)
        
        label = torch.tensor(picks).to(device)
    
    print(y_max)


        
        
        # all_ones = y.all(dim=1)
        # label = all_ones.squeeze(-1).to(torch.float32)
        # print(label)
        # print(label.shape)
        # for i in range(label.shape[0]):  # result.shape[0] is 20
        #     # Find indices where the value is 1 in the ith row
        #     indices = (label[i] == 1).nonzero(as_tuple=True)[0]
            
        #     # Print the indices for the ith row
        #     print(f"Row {i}: Indices with value 1: {indices.tolist()}") # empty now??