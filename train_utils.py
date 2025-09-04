import torch

def train_one_epoch(model, dataloader, criterion, optimizer, epoch, teacher_forcing_ratio):
    model.train()
    running_loss = 0.0
    for batch_idx, (src, src_lens, trg, trg_lens) in enumerate(dataloader):
        src, src_lens, trg, trg_lens = src.to(model.device), src_lens.to(model.device), trg.to(model.device), trg_lens.to(model.device)
        optimizer.zero_grad()
        outputs = model(src, src_lens, trg, trg_lens, teacher_forcing_ratio=teacher_forcing_ratio)
        out_dim = outputs.size(-1)
        loss = criterion(outputs[:, 1:, :].reshape(-1, out_dim), trg[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch}: train loss = {avg_loss:.4f}")
    return avg_loss