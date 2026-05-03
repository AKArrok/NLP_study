import torch
from tqdm import tqdm

import config
from dataset import get_dataloader
from model import ReviewAnalyzeModel
from predict import predict_batch


def evaluate_model(dataloader, model, device):
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc='评估'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        probs = predict_batch(input_ids, attention_mask, model)
        preds = [1 if p >= 0.5 else 0 for p in probs]

        for pred, label in zip(preds, labels):
            if pred == int(label.item()):
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f'准确率: {acc:.4f}')


def run_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = config.MODELS_DIR / 'model.pt'
    if not model_path.exists():
        raise FileNotFoundError(
            f'未找到模型文件: {model_path}\n'
            f'请先运行: python train.py'
        )
    model = ReviewAnalyzeModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    dataloader = get_dataloader(train=False)
    evaluate_model(dataloader, model, device)


if __name__ == '__main__':
    run_evaluate()
