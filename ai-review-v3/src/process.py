import json

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import config


def _resolve_pretrained_path():
    local_model_dir = config.PRE_TRAINED_DIR / 'bert-base-chinese'
    if local_model_dir.exists():
        return str(local_model_dir)
    return 'bert-base-chinese'


def process_data():
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 读取 CSV
    df = pd.read_csv(config.RAW_DATA_DIR / 'review.csv', encoding='utf-8-sig')
    df = df[['review', 'label']].dropna()
    df['review'] = df['review'].astype(str).str.strip()
    df = df[df['review'] != '']
    df['label'] = df['label'].astype(int)

    # 划分训练/测试
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label'],
    )
    print('数据集划分完成')

    tokenizer = AutoTokenizer.from_pretrained(_resolve_pretrained_path())

    def tokenize(text):
        encoded = tokenizer(
            text,
            max_length=config.SEQ_LEN,
            truncation=True,
            padding='max_length',
        )
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
        }

    def to_records(sub_df):
        records = []
        for _, row in sub_df.iterrows():
            encoded = tokenize(row['review'])
            records.append(
                {
                    'input_ids': encoded['input_ids'],
                    'attention_mask': encoded['attention_mask'],
                    'label': int(row['label']),
                }
            )
        return records

    train_records = to_records(train_df)
    test_records = to_records(test_df)
    print('分词编码完成')

    with open(config.PROCESSED_DATA_DIR / 'train.jsonl', 'w', encoding='utf-8') as f:
        for item in train_records:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(config.PROCESSED_DATA_DIR / 'test.jsonl', 'w', encoding='utf-8') as f:
        for item in test_records:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print('处理后的数据已保存')


if __name__ == '__main__':
    process_data()
