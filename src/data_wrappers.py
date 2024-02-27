from typing import List, Tuple, Optional, Union, Dict
import enum

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from lightning import LightningDataModule
from tqdm import tqdm
from tokenizers import Tokenizer


class Stages(enum.Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def preprocess(smiles: List[str], tokenizer):
    """
    Preprocess the data by tokenizing.
    """
    tokens = [tokenizer.encode(smi).ids for smi in tqdm(smiles)]
    tokens = [torch.tensor(i).long() for i in tokens]
    return tokens


class ZinkDatasetWrapper(Dataset):
    def __init__(self, zink_dataset, tokenizer: Tokenizer):
        super().__init__()
        smi_list = []

        for i in range(len(zink_dataset)):
            smi_list.append(zink_dataset[i]['smiles'])

        self.input_ids = preprocess(smi_list, tokenizer)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return self.input_ids[i]


class ZinkDataModule(LightningDataModule):
    def __init__(self,
                 pretrained_tokenizer_path,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 shuffle_train: bool = True):
        super().__init__()
        self.zink_dataset = load_dataset("sagawa/ZINC-canonicalized")
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers

        PAD_TOKEN = " "
        EOS_TOKEN = "&"

        self.tokenizer = Tokenizer.from_file(pretrained_tokenizer_path)
        self.tokenizer.eos_token = EOS_TOKEN
        self.tokenizer.pad_token = PAD_TOKEN

    def setup(self, stage: 'Optional[str]' = None) -> None:

        if stage == "fit" or stage is None:
            self.train = ZinkDatasetWrapper(self.zink_dataset['train'], self.tokenizer)
            self.val = ZinkDatasetWrapper(self.zink_dataset['validation'], self.tokenizer)
        if stage == "validate":
            self.val = ZinkDatasetWrapper(self.zink_dataset['validation'], self.tokenizer)
        if stage == "test":
            self.test = ZinkDatasetWrapper(self.zink_dataset['validation'], self.tokenizer)

    def collate_fn(self,
                   batch: List[
                       torch.Tensor
                   ]
                   ) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.token_to_id(self.tokenizer.pad_token)
        input_ids = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True,
                                                    padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(pad_id),
        )

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=self.shuffle_train)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)
