import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config


def process():
    """
    数据预处理主函数。
    """
    print('开始处理数据')

    # 读取中英文对齐数据
    df = pd.read_csv(
        config.RAW_DATA_DIR / 'cmn.txt',
        sep='\t',
        header=None,
        usecols=[0, 1],
        names=['en', 'zh']
    )

    # 清理空值数据
    df = df.dropna()
    df = df[df['en'].str.strip().ne('') & df['zh'].str.strip().ne('')]

    # 划分训练集和测试集
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    # 构建词表
    EnglishTokenizer.build_vocab(
        train_df['en'].tolist(),
        config.PROCESSED_DATA_DIR / 'en_vocab.txt'
    )
    ChineseTokenizer.build_vocab(
        train_df['zh'].tolist(),
        config.PROCESSED_DATA_DIR / 'zh_vocab.txt'
    )

    # 加载词表
    en_tokenizer = EnglishTokenizer.from_vocab(
        config.PROCESSED_DATA_DIR / 'en_vocab.txt'
    )
    zh_tokenizer = ChineseTokenizer.from_vocab(
        config.PROCESSED_DATA_DIR / 'zh_vocab.txt'
    )

    # 编码训练集
    train_df['en'] = train_df['en'].apply(
        lambda x: en_tokenizer.encode(
            x, seq_len=config.SEQ_LEN, add_sos_eos=True
        )
    )
    train_df['zh'] = train_df['zh'].apply(
        lambda x: zh_tokenizer.encode(
            x, seq_len=config.SEQ_LEN, add_sos_eos=False
        )
    )
    train_df.to_json(
        config.PROCESSED_DATA_DIR / 'indexed_train.jsonl',
        orient='records',
        lines=True
    )

    # 编码测试集
    test_df['en'] = test_df['en'].apply(
        lambda x: en_tokenizer.encode(
            x, seq_len=config.SEQ_LEN, add_sos_eos=True
        )
    )
    test_df['zh'] = test_df['zh'].apply(
        lambda x: zh_tokenizer.encode(
            x, seq_len=config.SEQ_LEN, add_sos_eos=False
        )
    )
    test_df.to_json(
        config.PROCESSED_DATA_DIR / 'indexed_test.jsonl',
        orient='records',
        lines=True
    )

    print('数据处理完成')


if __name__ == '__main__':
    process()
