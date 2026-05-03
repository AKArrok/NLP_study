import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import ReviewAnalyzeModel


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc='训练'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].float().to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    dataloader = get_dataloader(train=True)
    print(f'加载训练集完成, batch数: {len(dataloader)}')

    freeze_bert = bool(device.type == 'cpu' and config.FREEZE_BERT_ON_CPU)
    print(f'freeze_bert: {freeze_bert}')
    model = ReviewAnalyzeModel(freeze_bert=freeze_bert).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_function = torch.nn.BCEWithLogitsLoss()
    writer = SummaryWriter(
        log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S')
    )

    best_loss = float('inf')
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.EPOCHS + 1):
        print(f'========== Epoch {epoch} ==========')
        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            loss_function,
            device,
        )
        print(f'loss: {avg_loss:.4f}')
        writer.add_scalar('Loss/train', avg_loss, epoch)
        if epoch == 1 or avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print(f'已保存模型: {config.MODELS_DIR / "model.pt"}')
    writer.close()


if __name__ == '__main__':
    train()
