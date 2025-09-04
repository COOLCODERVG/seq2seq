import torch
import torch.nn as nn
import random
from vocab import INPUT_PAD_IDX, OUTPUT_PAD_IDX

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, emb_size, hidden_size, pad_idx=INPUT_PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, emb_size, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)

    def forward(self, src, src_lens):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, emb_size, hidden_size, pad_idx=OUTPUT_PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(output_vocab_size, emb_size, padding_idx=pad_idx)
        self.gru = nn.GRU(emb_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_vocab_size)

    def step(self, input_token, hidden):
        embedded = self.embedding(input_token).unsqueeze(1)
        out, hidden = self.gru(embedded, hidden)
        logits = self.linear(out.squeeze(1))
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_lens, trg, trg_lens, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_trg_len = trg.size(1)
        vocab_size = self.decoder.linear.out_features

        outputs = torch.zeros(batch_size, max_trg_len, vocab_size, device=self.device)
        hidden = self.encoder(src, src_lens)

        input_tok = trg[:, 0]
        for t in range(1, max_trg_len):
            logits, hidden = self.decoder.step(input_tok, hidden)
            outputs[:, t, :] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_tok = trg[:, t] if teacher_force else top1
        return outputs