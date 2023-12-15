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

class CombinedModel(nn.Module):
    def __init__(self, **vq_kwargs):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # changed input channel to 3
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # added padding
                nn.GELU(),
                nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1), # added padding
                VectorQuantize(dim=2, accept_image_fmap = True, **vq_kwargs),
            ]
        )

        self.model_pred = BERT(hidden = 756)
        
        return

    def forward(self, x):
        x, indices, commit_loss = self.encode(x)
        x = x.permute(0, 3, 1, 2)
        x = torch.nn.functional.pad(x.squeeze(), pad=(0,5), mode="constant", value=0)
        x = self.model_pred(x)

        return x

    def encode(self, x):
        # print(x)
        for layer in self.encoder:
            if isinstance(layer, VectorQuantize):
                x, indices, commit_loss = layer(x)
            else:
                x = layer(x)
        
        return x, indices, commit_loss


def crop_aligned_tensors(tensor1, tensor2, new_size=3000):
  """
  Crops two tensors with randomly chosen aligned window centers.

  Args:
    tensor1: First tensor of shape [8, 1, 9001, 2].
    tensor2: Second tensor of shape [8, 1, 9001, 2].
    new_size: Desired size of the cropped tensors (default: 3000).

  Returns:
    cropped_tensor1, cropped_tensor2: Cropped versions of the original tensors.
  """
  # Calculate the maximum window center offset for aligned cropping.
  max_offset = (tensor1.shape[2] - new_size)

  # Randomly choose the window center offset within the valid range.
  offset = torch.randint(0, max_offset + 1, (1,))

  # Calculate the start and end indices for cropping.
  start_index = offset[0]
  end_index = start_index + new_size

  # Crop both tensors using the same window.
  cropped_tensor1 = tensor1[:,:, start_index:end_index, :]
  cropped_tensor2 = tensor2[:, start_index:end_index,:, :]

  return cropped_tensor1, cropped_tensor2

if __name__ == '__main__':

    
    lr = 3e-4
    train_iter = 1000
    num_codes = 256
    seed = 1234
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch_num = 10
    
    data_dir = './dataset/waveform_train/'  
    data_list = './dataset/waveform.csv'


    dataset = SeismicDataset(data_dir = data_dir, data_list = data_list, transform=None)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = CombinedModel(codebook_size=num_codes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # loss = torch.nn.MSELoss()
    loss = torch.nn.CrossEntropyLoss()

    for _ in range(epoch_num):
        loss_accum = 0
        acc_accum = 0
        for i, (x, y) in enumerate(train_loader):
            print(f"step: {i}")
            optimizer.zero_grad()
            # out = torch.nn.functional.pad(x.squeeze(), pad=(0,5), mode="constant", value=0)
            x = x.permute(0, 3, 1, 2).to(device)
            x, y = crop_aligned_tensors(x, y)
            outputs = model(x)
            print(y.shape)
            labels = y[:, :, 0, 2]
            labels = to_gaussian_batch(labels).to(device)
            loss_output = loss(outputs, labels)
            loss_output.backward()


            optimizer.step()
            
            ground_truth = labels.max(dim=1)[1].numpy(force=True)
            outputs = outputs.squeeze(1)
            preds = outputs.max(dim=1)[1].numpy(force=True)
            gap = ground_truth - preds
            gap = numpy.absolute(gap)
            accuracy = len(gap[gap<50])/len(gap)

            loss_accum += loss_output.item()
            acc_accum +=  accuracy


        print(f"loss: {loss_accum/20:.3f}")
        print(f"accuracy: {acc_accum/20:.3f}")


