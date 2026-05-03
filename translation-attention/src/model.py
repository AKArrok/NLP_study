import torch
from torch import nn
from torchinfo import summary
import config


class Attention(nn.Module):
    """
    注意力机制模块：计算当前 decoder 状态与 encoder 输出的注意力上下文向量。
    """

    def forward(self, decoder_hidden, encoder_outputs):
        """
        计算注意力权重并加权求和生成上下文向量。
        :param decoder_hidden: 当前时间步解码器隐藏状态 (1, batch_size, decoder_hidden_dim)
        :param encoder_outputs: 编码器所有时间步输出 (batch_size, seq_len, decoder_hidden_dim)
        :return: 上下文向量 (batch_size, 1, decoder_hidden_dim)
        """
        # 计算注意力分数
        attention_scores = torch.bmm(
            decoder_hidden.transpose(0, 1),  # (batch_size, 1, hidden_dim)
            encoder_outputs.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        )
        attention_weights = torch.softmax(
            attention_scores, dim=2
        )  # (batch_size, 1, seq_len)

        # 加权求和，得到上下文向量
        context_vector = torch.bmm(
            attention_weights, encoder_outputs
        )  # (batch_size, 1, hidden_dim)
        return context_vector


class TranslationEncoder(nn.Module):
    """
    编码器模块：双向 GRU 编码中文句子。
    """

    def __init__(self, vocab_size, padding_index):
        """
        初始化编码器。
        :param vocab_size: 中文词表大小。
        :param padding_index: padding 索引。
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_index
        )
        self.rnn = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.ENCODER_HIDDEN_DIM,
            num_layers=config.ENCODER_LAYERS,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, src):
        """
        前向传播。
        :param src: 中文输入索引序列 (batch_size, seq_len)
        :return: (encoder_outputs, encoder_hidden)
        """
        embedded = self.embedding(src)  # (batch_size, seq_len, embedding_dim)
        output, hidden = self.rnn(embedded)
        return output, hidden


class TranslationDecoder(nn.Module):
    """
    解码器模块：单向 GRU + Attention，逐步生成英文翻译。
    """

    def __init__(self, vocab_size, padding_index):
        """
        初始化解码器。
        :param vocab_size: 英文词表大小。
        :param padding_index: padding 索引。
        """
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            padding_idx=padding_index
        )
        self.rnn = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.DECODER_HIDDEN_DIM,
            batch_first=True
        )
        # 输出维度是 hidden + context 拼接
        self.linear = nn.Linear(
            in_features=2 * config.DECODER_HIDDEN_DIM,
            out_features=vocab_size
        )
        self.attention = Attention()

    def forward(self, tgt, hidden, encoder_outputs):
        """
        前向传播。
        :param tgt: 当前输入 token，形状 (batch_size, 1)
        :param hidden: 上一时间步隐藏状态 (1, batch_size, hidden_dim)
        :param encoder_outputs: 编码器所有输出 (batch_size, seq_len, hidden_dim)
        :return: (output_logits, new_hidden)
        """
        embedded = self.embedding(tgt)  # (batch_size, 1, embedding_dim)
        output, hidden = self.rnn(
            embedded, hidden
        )  # output: (batch_size, 1, hidden_dim)
        context_vector = self.attention(
            hidden, encoder_outputs
        )  # (batch_size, 1, hidden_dim)

        # 拼接当前输出和上下文向量
        combined = torch.cat((output, context_vector), dim=2)
        output = self.linear(
            combined
        )  # (batch_size, 1, vocab_size)
        return output, hidden


if __name__ == '__main__':
    encoder = TranslationEncoder(vocab_size=10000, padding_index=0)
    dummy_encoder_input = torch.randint(
        low=0,
        high=10000,
        size=(config.BATCH_SIZE, config.SEQ_LEN)
    )
    summary(encoder, input_data=dummy_encoder_input)

    print('-' * 100)
    decoder = TranslationDecoder(vocab_size=10000, padding_index=0)
    dummy_decoder_input = torch.randint(
        low=0,
        high=10000,
        size=(config.BATCH_SIZE, 1)
    )
    dummy_decoder_hidden = torch.randn(
        size=(1, config.BATCH_SIZE, config.DECODER_HIDDEN_DIM)
    )
    dummy_encoder_outputs = torch.randn(
        size=(config.BATCH_SIZE, config.SEQ_LEN, config.DECODER_HIDDEN_DIM)
    )
    summary(
        decoder,
        input_data=[
            dummy_decoder_input,
            dummy_decoder_hidden,
            dummy_encoder_outputs
        ]
    )
