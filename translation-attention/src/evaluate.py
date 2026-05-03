import torch
from tqdm import tqdm
import config
from tokenizer import ChineseTokenizer, EnglishTokenizer
from model import TranslationEncoder, TranslationDecoder
from dataset import get_dataloader
from predict import predict_batch


def _ngram_counts(tokens, n):
    counts = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _corpus_bleu(references, hypotheses, max_n=4):
    """
    轻量 BLEU 实现，避免额外依赖。
    references: List[List[List[int]]]  每个样本的参考序列集合
    hypotheses: List[List[int]]        每个样本的预测序列
    """
    import math

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_length = 0
    hyp_length = 0

    for refs, hyp in zip(references, hypotheses):
        hyp_length += len(hyp)
        ref_lens = [len(ref) for ref in refs]
        # 选最接近长度的参考句
        ref_len = min(ref_lens, key=lambda l: (abs(l - len(hyp)), l))
        ref_length += ref_len

        for n in range(1, max_n + 1):
            hyp_counts = _ngram_counts(hyp, n)
            total_counts[n - 1] += sum(hyp_counts.values())

            max_ref_counts = {}
            for ref in refs:
                ref_counts = _ngram_counts(ref, n)
                for gram, c in ref_counts.items():
                    max_ref_counts[gram] = max(max_ref_counts.get(gram, 0), c)

            for gram, c in hyp_counts.items():
                clipped_counts[n - 1] += min(c, max_ref_counts.get(gram, 0))

    precisions = []
    for i in range(max_n):
        # add-1 平滑，避免 0 精度导致整体为 0
        precisions.append((clipped_counts[i] + 1) / (total_counts[i] + 1))

    if hyp_length == 0:
        return 0.0
    bp = 1.0 if hyp_length > ref_length else math.exp(1 - ref_length / hyp_length)
    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return score


def evaluate(dataloader, encoder, decoder, zh_tokenizer, en_tokenizer, device):
    """
    执行模型评估，计算 BLEU 分数。
    :param dataloader: 数据加载器。
    :param encoder: 编码器。
    :param decoder: 解码器。
    :param zh_tokenizer: 中文分词器。
    :param en_tokenizer: 英文分词器。
    :param device: 设备。
    :return: BLEU 分数。
    """
    all_references = []
    all_predictions = []
    special_tokens = [
        zh_tokenizer.pad_token_index,
        zh_tokenizer.eos_token_index,
        zh_tokenizer.sos_token_index
    ]

    for src, tgt in tqdm(dataloader, desc="评估"):
        src = src.to(device)
        tgt = tgt.tolist()
        predict_indexes = predict_batch(
            src, encoder, decoder, en_tokenizer, device
        )
        all_predictions.extend(predict_indexes)

        for indexes in tgt:
            indexes = [index for index in indexes if index not in special_tokens]
            all_references.append([indexes])

    bleu = _corpus_bleu(all_references, all_predictions)
    return bleu


def run_evaluate():
    """
    启动评估流程。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zh_tokenizer = ChineseTokenizer.from_vocab(
        config.PROCESSED_DATA_DIR / 'zh_vocab.txt'
    )
    en_tokenizer = EnglishTokenizer.from_vocab(
        config.PROCESSED_DATA_DIR / 'en_vocab.txt'
    )

    encoder = TranslationEncoder(
        zh_tokenizer.vocab_size,
        zh_tokenizer.pad_token_index
    ).to(device)
    encoder.load_state_dict(
        torch.load(config.MODELS_DIR / 'encoder.pt', map_location=device)
    )

    decoder = TranslationDecoder(
        en_tokenizer.vocab_size,
        en_tokenizer.pad_token_index
    ).to(device)
    decoder.load_state_dict(
        torch.load(config.MODELS_DIR / 'decoder.pt', map_location=device)
    )

    dataloader = get_dataloader(train=False)
    bleu = evaluate(
        dataloader, encoder, decoder, zh_tokenizer, en_tokenizer, device
    )
    print('========== 评估结果 ==========')
    print(f'BLEU: {bleu:.2f}')
    print('=============================')


if __name__ == '__main__':
    run_evaluate()
