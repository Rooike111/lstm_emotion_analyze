import torch
from model import ReviewAnalyzeModel
import config
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer

def evaluate(model,test_dataloader,device):
    total_count=0
    corrent_count=0
    for inputs,targets in test_dataloader:

        inputs =   inputs.to(device)
        targets = targets.tolist()
        batch_result = predict_batch(model,inputs)
        for result,target in zip(batch_result,targets):
            result = 1 if result >0.5 else 0 
            if result == target:
                corrent_count+=1
            total_count+=1

    return corrent_count/total_count
            


def run_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt'  )

    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size,padding_index=tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR/ 'best.pt'))
    print("模型加载成功")

    test_dataloader = get_dataloader(False)

    acc = evaluate(model,test_dataloader,device)
    
    print("评估结果")
    print(f"acc:{acc}")

if __name__ == "__main__":
    run_evaluate()