from abc import abstractmethod
from tqdm import tqdm


class BaseTokenizer:
    """
    分词器基类，提供词表构建、编码、解码等基础功能。
    """
    unk_token = '<unk>'
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'

    @staticmethod
    @abstractmethod
    def tokenize(sentence):
        """
        分词抽象方法。
        :param sentence: 输入句子。
        :return: 分词结果。
        """
        pass

    @abstractmethod
    def decode(self, indexes):
        """
        解码抽象方法。
        :param indexes: 索引列表。
        :return: 解码后的句子。
        """
        pass

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        """
        构建词表并保存。
        :param sentences: 句子列表。
        :param vocab_file: 保存词表文件路径。
        """
        unique_words = set()
        for sentence in tqdm(sentences, desc='分词'):
            for word in cls.tokenize(sentence):
                unique_words.add(word)

        vocab_list = [
            cls.pad_token,
            cls.unk_token,
            cls.sos_token,
            cls.eos_token,
        ] + list(unique_words)

        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')

    def __init__(self, vocab_list):
        """
        初始化分词器。
        :param vocab_list: 词表列表。
        """
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {
            word: idx for idx, word in enumerate(vocab_list)
        }
        self.index2word = {
            idx: word for idx, word in enumerate(vocab_list)
        }
        self.unk_token_index = self.word2index[self.unk_token]
        self.pad_token_index = self.word2index[self.pad_token]
        self.sos_token_index = self.word2index[self.sos_token]
        self.eos_token_index = self.word2index[self.eos_token]

    @classmethod
    def from_vocab(cls, vocab_file):
        """
        加载词表文件。
        :param vocab_file: 文件路径。
        :return: 分词器实例。
        """
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)

    def encode(self, sentence, seq_len, add_sos_eos=False):
        """
        将句子编码为索引列表。
        :param sentence: 输入句子。
        :param seq_len: 最大序列长度。
        :param add_sos_eos: 是否添加 <sos> 和 <eos>。
        :return: 索引列表。
        """
        tokens = self.tokenize(sentence)
        indexes = [
            self.word2index.get(token, self.unk_token_index)
            for token in tokens
        ]
        if add_sos_eos:
            indexes = indexes[:seq_len - 2]
            indexes = [self.sos_token_index] + indexes + [self.eos_token_index]
        else:
            indexes = indexes[:seq_len]

        if len(indexes) < seq_len:
            indexes += [self.pad_token_index] * (seq_len - len(indexes))
        return indexes


class ChineseTokenizer(BaseTokenizer):
    @staticmethod
    def tokenize(sentence):
        return list(sentence)

    def decode(self, indexes):
        return ''.join([self.index2word[index] for index in indexes])


class EnglishTokenizer(BaseTokenizer):
    @staticmethod
    def tokenize(sentence):
        from nltk import word_tokenize
        return word_tokenize(sentence)

    def decode(self, indexes):
        from nltk import TreebankWordDetokenizer
        tokens = [self.index2word[index] for index in indexes]
        return TreebankWordDetokenizer().detokenize(tokens)
