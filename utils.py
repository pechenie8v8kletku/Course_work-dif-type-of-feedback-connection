import os
import torch
from tqdm import tqdm

''' функция обучения одной эпохи'''
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, leave=False)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

'''

функция для валидация

'''
@torch.no_grad()
def validate(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        predicted = torch.argmax(pred, dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    val_acc = correct / total
        
    return total_loss / len(loader),val_acc

def save_epoch_checkpoint(model, optimizer, epoch, path):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, file_path)
def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
