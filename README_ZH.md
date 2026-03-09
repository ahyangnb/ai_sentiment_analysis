# 情感分析工具

[English](README.md) | [中文](README_zh.md)

一个本地运行的情感分析工具，可分析文本（推文、产品评论等）并判断情感倾向为**正面**、**负面**或**中性**。

完全离线运行，无需任何 API 令牌。

![示例图片](image.png)

## 技术栈

- **Python** + **Hugging Face Transformers**（用于三分类情感分析的 RoBERTa 模型）
- **NLTK VADER**（轻量级基于规则的情感分析器）
- **Gradio**（网页界面）

## 安装设置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate

# 安装依赖包
pip install -r requirements.txt
```

## 使用方法

### 命令行 - 单条文本
```bash
python main.py --text "这个产品太棒了！"
```

### 命令行 - 批量处理文件
```bash
python main.py --file texts.txt
```

### 命令行 - 使用 VADER 引擎（轻量级，无需下载模型）
```bash
python main.py --engine vader --text "体验太糟糕了"
```

### 命令行 - 交互模式
```bash
python main.py
```

### 网页界面
```bash
python main.py --web
```

## 分析引擎

| 引擎 | 模型 | 优势 |
|--------|-------|------|
| `transformer`（默认） | `cardiffnlp/twitter-roberta-base-sentiment-latest` | 准确率更高，基于深度学习 |
| `vader` | NLTK VADER | 速度快，无需GPU，无需下载大模型 |