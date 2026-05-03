import torch
import warnings

import config
from model import TranslationModel
from tokenizer import ChineseTokenizer, EnglishTokenizer

warnings.filterwarnings(
    'ignore',
    message='The PyTorch API of nested tensors is in prototype stage and will change in the near future.*',
    category=UserWarning,
)


def _generate_causal_mask(seq_len, device):
    # True means masked for bool attention masks in PyTorch Transformer.
    return torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=1,
    )


def predict_batch(input_tensor, model, en_tokenizer, device):
    model.eval()
    with torch.no_grad():
        src_pad_mask = input_tensor == model.src_embedding.padding_idx
        memory = model.encode(src=input_tensor, src_pad_mask=src_pad_mask)

        batch_size = input_tensor.shape[0]
        decoder_input = torch.full(
            size=(batch_size, 1),
            fill_value=en_tokenizer.sos_token_index,
            device=device,
        )
        generated = [[] for _ in range(batch_size)]
        finished = [False for _ in range(batch_size)]

        for _ in range(1, config.SEQ_LEN):
            tgt_mask = _generate_causal_mask(decoder_input.shape[1], device)
            tgt_pad_mask = decoder_input == en_tokenizer.pad_token_index
            decoder_output = model.decode(
                decoder_input,
                memory,
                tgt_mask,
                tgt_pad_mask,
                src_pad_mask,
            )
            predict_indexes = decoder_output[:, -1, :].argmax(dim=-1)

            for i in range(batch_size):
                if finished[i]:
                    continue
                token_id = predict_indexes[i].item()
                if token_id == en_tokenizer.eos_token_index:
                    finished[i] = True
                else:
                    generated[i].append(token_id)

            if all(finished):
                break
            decoder_input = torch.cat(
                [decoder_input, predict_indexes.unsqueeze(1)], dim=1
            )
        return generated


def predict(zh_sentence, model, zh_tokenizer, en_tokenizer, device):
    input_ids = zh_tokenizer.encode(
        zh_sentence, seq_len=config.SEQ_LEN, add_sos_eos=False
    )
    input_tensor = torch.tensor([input_ids], device=device)
    generated = predict_batch(input_tensor, model, en_tokenizer, device)
    en_indexes = generated[0]
    en_sentence = en_tokenizer.decode(en_indexes)
    return en_sentence


def run_predict():
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

    print('请输入中文句子（q 或 quit 退出）')
    while True:
        user_input = input('中文：')
        if user_input in ['q', 'quit']:
            print('谢谢使用，再见！')
            break
        if not user_input.strip():
            print('请输入内容')
            continue
        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print(f'英文：{result}')


if __name__ == '__main__':
    run_predict()
