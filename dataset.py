import random
import torch
from torch.utils.data import Dataset
import json
from vocab import INPUT_PAD_IDX, OUTPUT_PAD_IDX

digit_to_word = {
    '0': 'zero','1': 'one','2': 'two','3': 'three','4': 'four',
    '5': 'five','6': 'six','7': 'seven','8': 'eight','9': 'nine'
}

def make_example(min_len=1, max_len=8):
    L = random.randint(min_len, max_len)
    digits = ''.join(random.choice('0123456789') for _ in range(L))
    words = [digit_to_word[d] for d in digits]
    return digits, words

class DigitsWordsDataset(Dataset):
    def __init__(self, samples, input_idx, output_idx, sos_idx, eos_idx):
        self.samples = samples
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        digits, words = self.samples[idx]
        input_ids = [self.input_idx[d] for d in digits]
        output_ids = [self.sos_idx] + [self.output_idx[w] for w in words] + [self.eos_idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(output_ids, dtype=torch.long)

def save_dataset(samples, path="digits_dataset.json"):
    with open(path, "w") as f:
        json.dump(samples, f)

def load_dataset(path="digits_dataset.json"):
    with open(path, "r") as f:
        samples = json.load(f)
    return samples

def collate_fn(batch):
    inputs, outputs = zip(*batch)
    in_lens = [len(x) for x in inputs]
    out_lens = [len(y) for y in outputs]
    max_in = max(in_lens)
    max_out = max(out_lens)

    padded_in = torch.full((len(batch), max_in), INPUT_PAD_IDX, dtype=torch.long)
    padded_out = torch.full((len(batch), max_out), OUTPUT_PAD_IDX, dtype=torch.long)

    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        padded_in[i, :len(inp)] = inp
        padded_out[i, :len(out)] = out

    return padded_in, torch.tensor(in_lens), padded_out, torch.tensor(out_lens)