import torch, torchtext
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset, Subset

def create_text_loader(dataset, vocab, loader_kwargs):
    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for items in data_iter:
            yield tokenizer(items[-1])

    if vocab is None:
        vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
    else:
        pass

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return (text_list, offsets), label_list

    dataloader = DataLoader(dataset, collate_fn=collate_batch, **loader_kwargs)
    return dataloader, vocab

class IMDbDataset(Dataset):
    def __init__(self):
        self.filepath = "../dataset/IMDb/IMDB Dataset.csv"
        self.data_frame = pd.read_csv(self.filepath)
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        text = self.data_frame.iloc[idx, 0]  
        label = self.data_frame.iloc[idx, 1] 
        if label == "positive":
            label = 1
        elif label == "negative":
            label = 0
        else:
            raise ValueError("Label should be either 'positive' or 'negative'")

        return label, text

def split_IMDb_dataset(dataset):

    train_idx, test_idx = torch.load('../dataset/uciml_split_idx.pt')

    train_idx = torch.tensor(train_idx)
    test_idx = torch.tensor(test_idx)

    train_idx = train_idx[train_idx < len(dataset)]
    test_idx = test_idx[test_idx < len(dataset)]

    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    
    return train_dataset, test_dataset

def get_IMDb_dataset():
    dataset =  IMDbDataset()
    return split_IMDb_dataset(dataset)

if __name__ == "__main__":
    train_dataset, test_dataset = get_IMDb_dataset()
    print(train_dataset[0])
    print("Done")