import pandas as pd
import config
from sklearn.model_selection import train_test_split
from tokenizer import JiebaTokenizer


def process():
    print("开始处理数据...")

    df = pd.read_csv(config.RAW_DATA_DIR / "online_shopping_10_cats.csv", usecols = ["label","review"] ,encoding = 'utf-8' ).dropna()
    # print(df.head())

    # 划分数据集
    train_df,test_df = train_test_split(df,test_size = 0.2, stratify =df["label"])
    print("数据处理完成...")

    #构建词表
    JiebaTokenizer.build_vocab(train_df["review"].tolist(), config.MODELS_DIR / "vocab.txt")

    # 创建Tokenize
    tokenize = JiebaTokenizer.from_vocab(config.MODELS_DIR / "vocab.txt")

    # 计算序列长度
    # print(train_df["review"].apply(lambda x:len(tokenize.encode(x))).quantile(0.95)) 128

    #构建训练集与保存训练集
    train_df["review"] = train_df["review"].apply(lambda x:tokenize.encode(x))
    train_df.to_json(config.PROCESSED_DATA_DIR / "train.jsonl",orient='records',lines=True)
    
    #构建测试集与保存测试集
    test_df["review"] = test_df["review"].apply(lambda x:tokenize.encode(x))
    test_df.to_json(config.PROCESSED_DATA_DIR / "test.jsonl",orient='records',lines=True)

if __name__ == '__main__':
    process()