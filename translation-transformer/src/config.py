from pathlib import Path


# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 项目路径
MODELS_DIR = BASE_DIR / 'models'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
LOGS_DIR = BASE_DIR / 'logs'

# 模型参数
DIM_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

# 训练参数
BATCH_SIZE = 128
SEQ_LEN = 50
LEARNING_RATE = 1e-3
EPOCHS = 50
