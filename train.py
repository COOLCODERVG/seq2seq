import torch
from torch.utils.data import DataLoader, random_split
from dataset import load_dataset, DigitsWordsDataset, collate_fn
from models import Encoder, Decoder, Seq2Seq
from train_utils import train_one_epoch
from inference import greedy_decode
from vocab import INPUT_IDX, OUTPUT_IDX, IDX_OUTPUT, SOS_IDX, EOS_IDX, INPUT_VOCAB_SIZE, OUTPUT_VOCAB_SIZE, OUTPUT_PAD_IDX
from config import *

import torch.nn as nn
import random

def main():
    samples = load_dataset("digits_dataset.json")
    dataset = DigitsWordsDataset(samples, INPUT_IDX, OUTPUT_IDX, SOS_IDX, EOS_IDX)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    encoder = Encoder(INPUT_VOCAB_SIZE, EMB_SIZE, HIDDEN_SIZE).to(DEVICE)
    decoder = Decoder(OUTPUT_VOCAB_SIZE, EMB_SIZE, HIDDEN_SIZE).to(DEVICE)
    model = Seq2Seq(encoder, decoder, device=DEVICE).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=OUTPUT_PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, NUM_EPOCHS + 1):
        tf = max(0.1, 0.7 * (0.9 ** (epoch - 1)))
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, tf)
        print("Validation examples:")
        for ex in ["235", "007", "9", "31415", "48604"]:
            print(f"  {ex} -> {greedy_decode(model, ex, INPUT_IDX, IDX_OUTPUT, SOS_IDX, EOS_IDX)}")
        print("-" * 40)

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_vocab': INPUT_IDX,
        'output_vocab': OUTPUT_IDX
    }, "seq2seq_number2words.pth")

if __name__ == "__main__":
    main()
