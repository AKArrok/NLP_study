import jieba
from pathlib import Path

# Use a clean UTF-8 Chinese sample sentence for search-mode segmentation.
text = "小明硕士毕业于北京大学计算所"

words_generator = jieba.cut_for_search(text)
for word in words_generator:
    print(word)

words_list = jieba.lcut_for_search(text)
print(words_list)

user_dict_path = Path(__file__).resolve().parent / "dict.txt"
jieba.load_userdict(str(user_dict_path))
words_list = jieba.lcut("随着云计算技术的普及，越来越多企业开始采用云原生架构来部署服务，并借助大模型能力提升智能化水平，实现业务流程的自动化与智能决策。")
print(words_list)
