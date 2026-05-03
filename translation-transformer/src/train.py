import time

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import TranslationModel
from tokenizer import ChineseTokenizer, EnglishTokenizer


def _generate_causal_mask(seq_len, device):
    # True means masked for bool attention masks in PyTorch Transformer.
    return torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=1,
    )


def train_one_epoch(dataloader, model, loss_function, optimizer, device):
    model.train()
    total_loss = 0

    for src, tgt in tqdm(dataloader, desc='训练'):
        src = src.to(device)
        tgt = tgt.to(device)
        src_pad_mask = src == model.src_embedding.padding_idx
        tgt_pad_mask = tgt == model.tgt_embedding.padding_idx

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask = _generate_causal_mask(tgt_input.shape[1], device)

        optimizer.zero_grad()
        output = model(
            src,
            tgt_input,
            src_pad_mask,
            tgt_pad_mask[:, :-1],
            tgt_mask,
        )
        loss = loss_function(
            output.reshape(-1, output.shape[-1]),
            tgt_output.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataloader()
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.pad_token_index,
        en_tokenizer.pad_token_index,
    ).to(device)

    loss_function = CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print(f'========== Epoch {epoch} ==========')
        avg_loss = train_one_epoch(dataloader, model, loss_function, optimizer, device)
        print(f'平均损失: {avg_loss:.4f}')
        writer.add_scalar('Loss', avg_loss, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print('已保存模型')
        else:
            print('未保存模型')


if __name__ == '__main__':
    train()
