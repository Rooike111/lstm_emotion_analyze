# lstm_emotion_analyze 项目启动与运行说明

## 1. 项目说明
本项目基于 `LSTM + jieba` 做中文文本情感二分类，主要脚本位于 `src/`：
- `process.py`：原始数据预处理（分词、构建词表、生成训练/测试集）
- `train.py`：模型训练并保存最佳权重
- `predict.py`：加载训练好的模型进行交互式预测
- `evaluate.py`：加载测试集并计算模型准确率（Accuracy）

**此项目说明由ChatGPT 5.3生成**

## 2. 环境准备
建议 Python 3.10+。

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install torch pandas scikit-learn jieba tqdm tensorboard
```

## 3. 启动与运行命令
以下命令都在项目根目录执行。

### 3.1 数据预处理（首次训练前必须执行）
```powershell
python src\process.py
```
说明：
- 读取 `data/raw/online_shopping_10_cats.csv`
- 生成词表 `models/vocab.txt`
- 生成训练集与测试集：
  - `data/processed/train.jsonl`
  - `data/processed/test.jsonl`

### 3.2 启动训练
```powershell
python src\train.py
```
说明：
- 自动使用 CUDA（若可用），否则使用 CPU
- 训练日志写入 `logs/时间戳目录/`
- 最优模型保存为 `models/best.pt`

### 3.3 运行预测（交互模式）
```powershell
python src\predict.py
```
说明：
- 启动后输入一句中文文本并回车，返回情感预测概率
- 输入 `q` 或 `quit` 可退出

### 3.4 运行模型评估
```powershell
python src\evaluate.py
```
说明：
- 加载 `models/best.pt` 作为评估模型
- 使用 `data/processed/test.jsonl` 对模型进行准确率评估

## 4. 评估结果
当前模型测试集准确率（Accuracy）：

**91.49%**

## 5. 可选：查看训练曲线
训练过程中可在另一个终端运行：

```powershell
tensorboard --logdir logs
```
然后在浏览器打开命令行给出的本地地址（通常是 `http://localhost:6006`）。

## 6. 一次性完整流程
如果你从零开始，可按下面顺序执行：

```powershell
python src\process.py
python src\train.py
python src\evaluate.py
python src\predict.py
```
