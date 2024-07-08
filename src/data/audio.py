import os, torch, torchaudio
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchaudio.datasets import SPEECHCOMMANDS

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("../dataset", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

SC_labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
SC_sample_rate = 16000
SC_new_sample_rate = 1000

SC_downsample = torchaudio.transforms.Resample(orig_freq=SC_sample_rate, new_freq=SC_new_sample_rate)

def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(SC_labels.index(word))

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1).squeeze(1)
    batch = SC_downsample(batch)
    return batch

def SC_collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def get_SC_loader(dataset, loader_kwargs):
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=SC_collate_fn,
        pin_memory=True,
        **loader_kwargs,
    )

class SpeechCommandsDataset(Dataset):
    def __init__(self, subset: str = None):
        self.dataset = SubsetSC(subset)
        self.labels = SC_labels
        self.num_classes = len(self.labels)

    def __getitem__(self, index):
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[index]
        waveform = SC_downsample(waveform)
        waveform = waveform.squeeze(0)
        waveform = torch.nn.functional.pad(waveform, (0, 1000 - waveform.size(0)), 'constant', 0)
        label = label_to_index(label)
        return waveform, label

    def __len__(self):
        return len(self.dataset)

class UrbanSoundDataset(Dataset):

    def __init__(self):
        self.annotations = pd.read_csv("../dataset/UrbanSound8K/metadata/UrbanSound8K.csv")
        self.audio_dir = "../dataset/UrbanSound8K/audio"
        self.target_sample_rate = 1000
        self.num_samples = 1000

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        # signal -> (num_channels,samples) -> (2,16000) -> (1,16000)
        signal = self._resample_if_necessary(signal,sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        return signal.squeeze(0), label
    
    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples) -> (1,50000) -> (1,22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # [1,1,1] -> [1,1,1,0,0]
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: # (2,16000)
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


    def _get_audio_sample_path(self, index):
        if isinstance(index, int):
            pass
        elif isinstance(index, torch.Tensor):
            index = index.item()
        else:
            raise ValueError("index must be an integer or a tensor")
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        if isinstance(index, int):
            pass
        elif isinstance(index, torch.Tensor):
            index = index.item()
        else:
            raise ValueError("index must be an integer or a tensor")
        return self.annotations.iloc[index, 6]

def split_US_dataset(dataset):

    train_idx, test_idx = torch.load('../dataset/uciml_split_idx.pt')

    train_idx = torch.tensor(train_idx)
    test_idx = torch.tensor(test_idx)

    train_idx = train_idx[train_idx < len(dataset)]
    test_idx = test_idx[test_idx < len(dataset)]

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    return train_dataset, test_dataset

def get_US_dataset():
    dataset =  UrbanSoundDataset()
    return split_US_dataset(dataset)

if __name__ == "__main__":
    import tqdm

    def processing_dataset(dataset):
        x,y = [],[]
        for i in tqdm.tqdm(range(len(dataset))):
            item = dataset[i]
            x.append(item[0])
            y.append(item[1])
        x = torch.stack(x)
        y = torch.tensor(y).long()
        return TensorDataset(x,y)

    os.chdir("../")
    train_sc, test_sc = SpeechCommandsDataset("training"), SpeechCommandsDataset("testing")
    print(f"Num of samples: {len(train_sc)}, {len(test_sc)}, input shape: {train_sc[0][0].shape}")
    train_sc_processed, test_sc_processed = processing_dataset(train_sc), processing_dataset(test_sc)
    torch.save(train_sc_processed, "../dataset/SpeechCommands/train_sc.pt")
    torch.save(test_sc_processed, "../dataset/SpeechCommands/test_sc.pt")

    train_us, test_us = get_US_dataset()
    print(f"Num of samples: {len(train_us)}, {len(test_us)}, input shape: {train_us[0][0].shape}")
    train_us_processed, test_us_processed = processing_dataset(train_us), processing_dataset(test_us)
    torch.save(train_us_processed, "../dataset/UrbanSound8K/train_us.pt")
    torch.save(test_us_processed, "../dataset/UrbanSound8K/test_us.pt")
