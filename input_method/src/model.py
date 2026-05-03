import torch
from torch import nn
import config
#from torchinfo import summary



class InputMethodModel(nn.Module):
    """
    输入法预测模型，基于 RNN 的序列模型。
    """

    def __init__(self, vocab_size):
        """
        初始化模型。
        :param vocab_size: 词表大小。
        """
        super().__init__()

        # 嵌入层：将 token 索引映射为稠密向量
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
        )

        # RNN：处理序列数据，提取上下文特征
        self.rnn = nn.RNN(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            batch_first=True,
        )

        # 全连接层：将隐藏状态映射到词表大小的概率分布
        self.linear = nn.Linear(
            in_features=config.HIDDEN_SIZE,
            out_features=vocab_size,
        )

    def forward(self, x):
        """
        前向传播。
        :param x: 输入张量，形状 (batch_size, seq_len)。
        :return: 模型输出，形状 (batch_size, vocab_size)。
        """
        # 嵌入层处理输入序列
        embed = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        # RNN 处理嵌入向量序列
        output, _ = self.rnn(embed)  # (batch_size, seq_len, hidden_size)

        # 取最后一个时间步的输出进行分类
        result = self.linear(output[:, -1, :])  # (batch_size, vocab_size)
        return result


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InputMethodModel(vocab_size=20000).to(device)

    # 创建随机 dummy 输入用于展示模型结构
    dummy_input = torch.randint(
        low=0,
        high=20000,
        size=(config.BATCH_SIZE, config.SEQ_LEN),
        dtype=torch.long,
        device=device,
    )

