import pandas as pd
from sklearn.model_selection import train_test_split

import config
from tokenizer import ChineseTokenizer, EnglishTokenizer


def process():
    print('开始处理数据')
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        config.RAW_DATA_DIR / 'cmn.txt',
        sep='\t',
        header=None,
        usecols=[0, 1],
        names=['en', 'zh'],
    )
    df = df.dropna()
    df = df[df['en'].str.strip().ne('') & df['zh'].str.strip().ne('')]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    EnglishTokenizer.build_vocab(
        train_df['en'].tolist(), config.PROCESSED_DATA_DIR / 'en_vocab.txt'
    )
    ChineseTokenizer.build_vocab(
        train_df['zh'].tolist(), config.PROCESSED_DATA_DIR / 'zh_vocab.txt'
    )

    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')

    train_df['en'] = train_df['en'].apply(
        lambda x: en_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=True)
    )
    train_df['zh'] = train_df['zh'].apply(
        lambda x: zh_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=False)
    )
    train_df.to_json(
        config.PROCESSED_DATA_DIR / 'indexed_train.jsonl',
        orient='records',
        lines=True,
    )

    test_df['en'] = test_df['en'].apply(
        lambda x: en_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=True)
    )
    test_df['zh'] = test_df['zh'].apply(
        lambda x: zh_tokenizer.encode(x, seq_len=config.SEQ_LEN, add_sos_eos=False)
    )
    test_df.to_json(
        config.PROCESSED_DATA_DIR / 'indexed_test.jsonl',
        orient='records',
        lines=True,
    )
    print('数据处理完成')


if __name__ == '__main__':
    process()
