import math

import torch
from torch import nn

import config


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pos = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / self.d_model)
        )
        sins = torch.sin(pos / div_term)
        coss = torch.cos(pos / div_term)
        pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float)
        pe[:, 0::2] = sins
        pe[:, 1::2] = coss
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len]


class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        self.src_embedding = nn.Embedding(
            num_embeddings=zh_vocab_size,
            embedding_dim=config.DIM_MODEL,
            padding_idx=zh_padding_index,
        )
        self.tgt_embedding = nn.Embedding(
            num_embeddings=en_vocab_size,
            embedding_dim=config.DIM_MODEL,
            padding_idx=en_padding_index,
        )
        self.position_encoding = PositionEncoding(d_model=config.DIM_MODEL)
        self.transformer = nn.Transformer(
            d_model=config.DIM_MODEL,
            nhead=config.NUM_HEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            batch_first=True,
        )
        self.linear = nn.Linear(config.DIM_MODEL, en_vocab_size)

    def encode(self, src, src_pad_mask):
        src_embed = self.src_embedding(src)
        src_embed = self.position_encoding(src_embed)
        memory = self.transformer.encoder(
            src=src_embed,
            src_key_padding_mask=src_pad_mask,
        )
        return memory

    def decode(self, tgt, memory, tgt_mask, tgt_pad_mask, src_pad_mask):
        tgt_embed = self.tgt_embedding(tgt)
        tgt_embed = self.position_encoding(tgt_embed)
        output = self.transformer.decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.linear(output)

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask, tgt_mask):
        memory = self.encode(src, src_pad_mask)
        output = self.decode(tgt, memory, tgt_mask, tgt_pad_mask, src_pad_mask)
        return output
