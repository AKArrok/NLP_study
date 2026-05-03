import torch
from transformers import AutoTokenizer

import config
from model import ReviewAnalyzeModel


def _resolve_pretrained_path():
    local_model_dir = config.PRE_TRAINED_DIR / 'bert-base-chinese'
    if local_model_dir.exists():
        return str(local_model_dir)
    return 'bert-base-chinese'


def predict_batch(input_ids, attention_mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits)
        return probs.tolist()


def predict_text(user_input, model, tokenizer, device):
    encoded = tokenizer(
        user_input,
        max_length=config.SEQ_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    prob = predict_batch(input_ids, attention_mask, model)[0]
    return prob


def run_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(_resolve_pretrained_path())
    model = ReviewAnalyzeModel().to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))

    print('请输入评论（输入 q 或 quit 退出）')
    while True:
        user_input = input('> ').strip()
        if user_input.lower() in {'q', 'quit'}:
            print('已退出')
            break
        if not user_input:
            print('请输入内容')
            continue
        result = predict_text(user_input, model, tokenizer, device)
        if result > 0.5:
            print(f'正向评论（置信度: {result:.2f}）')
        else:
            print(f'负向评论（置信度: {result:.2f}）')


if __name__ == '__main__':
    run_predict()
