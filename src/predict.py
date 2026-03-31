import torch
from tokenizer import JiebaTokenizer
import config
from model import ReviewAnalyzeModel

def predict_batch(model,inputs):
    """
    返回预测结果
    """
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        #output.shape:[batch_size]
        batch_result = torch.sigmoid(output)
    return batch_result.tolist()
    

def predict(text,model,tokenizer,device):
    indexex = tokenizer.encode(text,seq_len = config.SEQ_LEN)
    input_tensor = torch.tensor([indexex],dtype = torch.long).to(device)
    batch_result = predict_batch(model=model,inputs=input_tensor)
    return batch_result[0]


def run_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    tokenizer = JiebaTokenizer.from_vocab(config.MODELS_DIR / 'vocab.txt')

    model = ReviewAnalyzeModel(vocab_size=tokenizer.vocab_size,padding_index = tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))
    print("模型加载成功")

    print("欢迎情感分析模型(输入q或quit退出)")

    while True:
        user_input = input("> ")
        if user_input in ['q','quit']:
            print("欢迎下次再来")
        if user_input.strip() == "":
            print("请重新输入内容")
            continue

        result = predict(model=model,tokenizer=tokenizer,device=device,text=user_input)
        if result > 0.5:
            print(f"正向(置信度:{result})")
        else:
            print(f"反向(置信度:{1-result})")

if __name__=="__main__":
    run_predict()
