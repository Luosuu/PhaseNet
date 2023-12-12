import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
# import obspy
import pandas as pd
import os

# from transformers import 

class DataConfig:
    seed = 123
    use_seed = True
    n_channel = 3
    n_class = 3
    sampling_rate = 100
    dt = 1.0 / sampling_rate
    X_shape = [3000, 1, n_channel]
    Y_shape = [3000, 1, n_class]
    min_event_gap = 3 * sampling_rate
    label_shape = "gaussian"
    label_width = 30
    dtype = "float32"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class SeismicDataset(Dataset):
    def __init__(
        self, 
        config = DataConfig(),
        format: str = "numpy",
        data_dir: str = './dataset/waveform_train/', 
        data_list: str = './dataset/waveform.csv',
        response_xml=None, 
        sampling_rate=100, 
        highpass_filter=0,
        transform = None, 
        **kwargs
        ):
        

        self.buffer = {}
        self.n_channel = config.n_channel
        self.n_class = config.n_class
        self.X_shape = config.X_shape
        self.Y_shape = config.Y_shape
        self.dt = config.dt
        self.dtype = config.dtype
        self.label_shape = config.label_shape
        # self.label_shape = "guassian"
        self.label_width = config.label_width
        self.config = config
        # if response_xml is not None:
        #     self.response = obspy.read_inventory(response_xml)
        # else:
        self.response = None
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.data_dir = data_dir
        self.data_list = data_list
        try:
            csv = pd.read_csv(self.data_list, header=0, sep="[,|\s+]", engine="python")
        except:
            csv = pd.read_csv(self.data_list, header=0, sep="\t")

        self.data_files = csv["fname"]
        self.data_paths = [os.path.join(self.data_dir, data_file) for data_file in self.data_files]
        self.data_full = [self.load_data(file) for file in self.data_paths]
        self.data = np.array([self.load_data(file)["data"] for file in self.data_paths])
        print(self.data.shape)
        self.data = self.data.reshape(154*9001,3)
        self.min_vals = self.data.min(axis=0)
        self.max_vals = self.data.max(axis=0)
        print(self.min_vals)
        print(self.max_vals)
        self.norm_data = (self.data - self.min_vals)/ (self.max_vals - self.min_vals)
        self.data = self.norm_data
        self.num_data = len(self.data_list)
        self.data = self.data.reshape(154,9001,1,3)
        for i, data in enumerate(self.data_full):
            self.data_full[i]["data"] = self.data[i]

    def load_data(self, fname):
        if fname not in self.buffer:
            npz = np.load(fname)
            meta = {}
            if len(npz["data"].shape) == 2:
                meta["data"] = npz["data"][:, np.newaxis, :]
                # Calculating the minimum and maximum values for each of the 3 columns 
                # min_vals = meta["data"].min(axis=0) 
                # max_vals = meta["data"].max(axis=0)
                # meta["data"] = (meta["data"] - min_vals) / (max_vals - min_vals)
                # print(min_vals)
                # print(max_vals)
                # print(meta["data"])
            else:
                meta["data"] = npz["data"]
            if "p_idx" in npz.files:
                if len(npz["p_idx"].shape) == 0:
                    meta["itp"] = [[npz["p_idx"]]]
                else:
                    meta["itp"] = npz["p_idx"]
            if "s_idx" in npz.files:
                if len(npz["s_idx"].shape) == 0:
                    meta["its"] = [[npz["s_idx"]]]
                else:
                    meta["its"] = npz["s_idx"]
            if "itp" in npz.files:
                if len(npz["itp"].shape) == 0:
                    meta["itp"] = [[npz["itp"]]]
                else:
                    meta["itp"] = npz["itp"]
            if "its" in npz.files:
                if len(npz["its"].shape) == 0:
                    meta["its"] = [[npz["its"]]]
                else:
                    meta["its"] = npz["its"]
            if "station_id" in npz.files:
                meta["station_id"] = npz["station_id"]
            if "sta_id" in npz.files:
                meta["station_id"] = npz["sta_id"]
            if "t0" in npz.files:
                meta["t0"] = npz["t0"]
            self.buffer[fname] = meta
        else:
            meta = self.buffer[fname]
        return meta

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        meta = self.data_full[idx]
        sample_data = np.copy(meta["data"])
        itp_list = meta["itp"]
        its_list = meta["its"]
        label = self.generate_label(sample_data, [itp_list, its_list])

        if self.transform is not None:
            sample_data = self.transform(sample_data.astype(np.float32))
        else:
            sample_data = sample_data.astype(np.float32)

        return sample_data, label.astype(np.float32)


    def generate_label(self, data, phase_list, mask=None):
        target = np.zeros_like(data)

        if self.label_shape == "gaussian":
            label_window = np.exp(
                -((np.arange(-self.label_width // 2, self.label_width // 2 + 1)) ** 2)
                / (2 * (self.label_width / 5) ** 2)
            )
        elif self.label_shape == "triangle":
            label_window = 1 - np.abs(
                2 / self.label_width * (np.arange(-self.label_width // 2, self.label_width // 2 + 1))
            )
        else:
            print(f"Label shape {self.label_shape} should be guassian or triangle")
            raise

        for i, phases in enumerate(phase_list):
            for j, idx_list in enumerate(phases):
                for idx in idx_list:
                    if np.isnan(idx):
                        continue
                    idx = int(idx)
                    if (idx - self.label_width // 2 >= 0) and (idx + self.label_width // 2 + 1 <= target.shape[0]):
                        target[idx - self.label_width // 2 : idx + self.label_width // 2 + 1, j, i + 1] = label_window

            target[..., 0] = 1 - np.sum(target[..., 1:], axis=-1)
            if mask is not None:
                target[:, mask == 0, :] = 0

        return target


if __name__ == '__main__':
    data_dir = './dataset/waveform_train/'  
    data_list = './dataset/waveform.csv'
    dataset = SeismicDataset(data_dir = data_dir, data_list = data_list)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for idx, batch in enumerate(data_loader):
        inputs, labels = batch
        collapsed_inputs = torch.squeeze(inputs, 2)
        collapsed_labels = torch.squeeze(labels, 2)
        if idx == 0:
            print(inputs)
            print(inputs.shape) # torch.Size([32, 9001, 1, 3]) [batch_size, sequence_length, ]
            print(labels)
            print(labels.shape) # torch.Size([32, 9001, 1, 3])
            print(collapsed_inputs.shape)
            print(collapsed_labels.shape)
