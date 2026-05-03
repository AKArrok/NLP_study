import math

import torch
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import TranslationModel
from predict import predict_batch
from tokenizer import ChineseTokenizer, EnglishTokenizer


def _ngram_counts(tokens, n):
    counts = {}
    if len(tokens) < n:
        return counts
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _corpus_bleu_fallback(references, hypotheses, max_n=4):
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_length = 0
    hyp_length = 0

    for refs, hyp in zip(references, hypotheses):
        hyp_length += len(hyp)
        ref_lens = [len(ref) for ref in refs]
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

    precisions = [(clipped_counts[i] + 1) / (total_counts[i] + 1) for i in range(max_n)]
    if hyp_length == 0:
        return 0.0
    bp = 1.0 if hyp_length > ref_length else math.exp(1 - ref_length / hyp_length)
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def evaluate(dataloader, model, zh_tokenizer, en_tokenizer, device):
    all_references = []
    all_predictions = []
    special_tokens = [
        zh_tokenizer.pad_token_index,
        zh_tokenizer.eos_token_index,
        zh_tokenizer.sos_token_index,
    ]
    for src, tgt in tqdm(dataloader, desc='评估'):
        src = src.to(device)
        tgt = tgt.tolist()
        predict_indexes = predict_batch(src, model, en_tokenizer, device)
        all_predictions.extend(predict_indexes)
        for indexes in tgt:
            indexes = [index for index in indexes if index not in special_tokens]
            all_references.append([indexes])

    return _corpus_bleu_fallback(all_references, all_predictions)


def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.pad_token_index,
        en_tokenizer.pad_token_index,
    ).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))
    dataloader = get_dataloader(train=False)
    bleu = evaluate(dataloader, model, zh_tokenizer, en_tokenizer, device)
    print(f'BLEU: {bleu:.2f}')


if __name__ == '__main__':
    run_evaluate()
