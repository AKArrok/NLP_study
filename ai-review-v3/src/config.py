from pathlib import Path


# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 路径设置
MODELS_DIR = BASE_DIR / 'models'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
LOGS_DIR = BASE_DIR / 'logs'
PRE_TRAINED_DIR = BASE_DIR / 'pretrained'

# 训练超参数
SEQ_LEN = 128
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
EPOCHS = 3
FREEZE_BERT_ON_CPU = True
