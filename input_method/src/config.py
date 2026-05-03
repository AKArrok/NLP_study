from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 数据路径
RAW_DATA_DIR = ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = ROOT_DIR / 'data' / 'processed'

# 模型和日志路径
MODELS_DIR = ROOT_DIR / 'models'
LOG_DIR = ROOT_DIR / 'logs'

# 训练参数
SEQ_LEN = 5 # 输入序列长度
BATCH_SIZE = 64 # 批大小
EMBEDDING_DIM = 64 # 嵌入层维度
HIDDEN_SIZE = 128 # RNN 隐藏层维度
LEARNING_RATE = 1e-3 # 学习率
EPOCHS = 3 # 训练轮数
TRAIN_SPLIT = 0.9 # 训练集比例
MAX_SENTENCES = 3000 # 预处理时最多使用的语句数
