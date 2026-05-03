import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
from tokenizer import JiebaTokenizer


def load_sentences():
    """
    加载原始语料。
    优先读取 data/raw，其次回退到 NLP 根目录下的 online_shopping_10_cats.csv。
    """
    csv_files = sorted(config.RAW_DATA_DIR.glob('*.csv'))
    txt_files = sorted(config.RAW_DATA_DIR.glob('*.txt'))

    if csv_files:
        source = csv_files[0]
    elif txt_files:
        source = txt_files[0]
    else:
        fallback = config.ROOT_DIR.parent / 'online_shopping_10_cats.csv'
        if not fallback.exists():
            raise FileNotFoundError('未找到原始语料，请在 data/raw 中放入 txt/csv 文件。')
        source = fallback

    print(f'使用语料文件: {source}')
    if source.suffix.lower() == '.csv':
        df = pd.read_csv(source)
        if 'review' not in df.columns:
            raise ValueError('CSV 文件缺少 review 列，无法提取文本语料。')
        sentences = [
            str(text).strip()
            for text in df['review'].tolist()
            if isinstance(text, str) and str(text).strip()
        ]
    else:
        sentences = [
            line.strip()
            for line in source.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    return sentences[: config.MAX_SENTENCES]


def build_samples(sentences, tokenizer):
    """
    生成训练样本。
    每个样本由长度为 SEQ_LEN 的输入序列和下一个词的目标索引组成。
    """
    samples = []
    for sentence in tqdm(sentences, desc='构建样本'):
        token_ids = tokenizer.encode(sentence)
        if len(token_ids) <= config.SEQ_LEN:
            continue

        for idx in range(len(token_ids) - config.SEQ_LEN):
            samples.append(
                {
                    'input': token_ids[idx : idx + config.SEQ_LEN],
                    'target': token_ids[idx + config.SEQ_LEN],
                }
            )

    return samples


def save_jsonl(samples, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def process():
    """
    数据预处理主函数。
    """
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences()
    if not sentences:
        raise ValueError('没有可用语料，无法继续预处理。')

    vocab_file = config.PROCESSED_DATA_DIR / 'vocab.txt'
    JiebaTokenizer.build_vocab(sentences, vocab_file)
    tokenizer = JiebaTokenizer.from_vocab(vocab_file)

    samples = build_samples(sentences, tokenizer)
    if not samples:
        raise ValueError('样本数为 0，请检查语料内容或 SEQ_LEN 配置。')

    split_index = int(len(samples) * config.TRAIN_SPLIT)
    train_samples = samples[:split_index]
    test_samples = samples[split_index:]

    save_jsonl(train_samples, config.PROCESSED_DATA_DIR / 'indexed_train.jsonl')
    save_jsonl(test_samples, config.PROCESSED_DATA_DIR / 'indexed_test.jsonl')

    print(f'语句数: {len(sentences)}')
    print(f'词表大小: {tokenizer.vocab_size}')
    print(f'训练样本数: {len(train_samples)}')
    print(f'测试样本数: {len(test_samples)}')
    print('预处理完成！')


if __name__ == '__main__':
    process()
