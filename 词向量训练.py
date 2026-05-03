import jieba
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent
csv_path = base_dir / 'online_shopping_10_cats.csv'
vector_path = base_dir / 'my_vectors.kv'

df = pd.read_csv(csv_path, encoding='utf-8', usecols=['review'])
missing_reviews = df["review"].isna().sum()
print(f"Missing reviews: {missing_reviews}")

reviews = df["review"].dropna().astype(str)

sentences = [
    [token for token in jieba.lcut(review) if token.strip() != ""]
    for review in reviews
]

model = Word2Vec(
sentences, # 已分词的句子序列
vector_size=100, # 词向量维度
window=5, # 上下文窗口大小
min_count=2, # 最小词频（低于将被忽略）
sg=1, # 1 = Skip-Gram，0 = CBOW
workers=4 # 并行训练线程数
)
model.wv.save_word2vec_format(vector_path)

my_model = KeyedVectors.load_word2vec_format(vector_path)
print(my_model)
