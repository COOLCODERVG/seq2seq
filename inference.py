import torch
from config import DEVICE, MAX_DECODING_STEPS

def greedy_decode(model, src_str, input_idx, idx_output, sos_idx, eos_idx):
    model.eval()
    max_words = len(src_str)
    if max_words == 0:
        return ""

    with torch.no_grad():
        src_ids = torch.tensor([[input_idx[c] for c in src_str]], dtype=torch.long, device=DEVICE)
        src_lens = torch.tensor([len(src_str)], dtype=torch.long, device=DEVICE)

        hidden = model.encoder(src_ids, src_lens)
        input_tok = torch.tensor([sos_idx], dtype=torch.long, device=DEVICE)

        out_words = []
        for step in range(max_words):
            logits, hidden = model.decoder.step(input_tok, hidden)

            if step < max_words - 1:
                logits[:, eos_idx] = -1e9 

            next_tok = logits.argmax(1).item()

            if next_tok in (sos_idx, eos_idx):
                topk = torch.topk(logits, k=3, dim=1).indices.squeeze(0).tolist()
                for alt in topk:
                    if alt not in (sos_idx, eos_idx):
                        next_tok = alt
                        break

            out_words.append(idx_output[next_tok])
            input_tok = torch.tensor([next_tok], dtype=torch.long, device=DEVICE)

        out_words = [tok for tok in out_words if tok not in ("<eos>", "<sos>", "<pad>")]
        return ' '.join(out_words)