# Function: 中英文混合处理工具
# Author: 刘琦

# !pip install jieba

import re
import nltk
import jieba
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 首次需要科学上网运行下面这行代码下载停用词表
# nltk.download('stopwords')  

def to_keywords(input_string):
    """将句子转成检索关键词序列"""
    # 按搜索引擎模式分词
        
    chinese_tokens = jieba.cut_for_search(re.sub(r'[a-zA-Z0-9\.\,\!\?\;\:\(\)\。\，\！\？\；\：\（\）\'\"\‘\’\“\”]', ' ', input_string))
    english_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9\s]', ' ', input_string))# 使用正则表达式替换所有非字母数字的字符为空格
    word_tokens = list(chinese_tokens) + english_tokens
    # word_tokens = list(chinese_tokens)
    # 加载中英文停用词表
    stop_words = set(stopwords.words('chinese')) | set(stopwords.words('english'))
    # 去除停用词
    filtered_sentence = [w for w in word_tokens if not (w in stop_words or w == ' ')]
    return ' '.join(filtered_sentence)

def sent_tokenize(input_string):
    """按标点断句"""
    # 按标点切分
    sentences = re.split(r'(?<=[。！？；?!.;])', input_string)
    # 去掉空字符串
    return [sentence.strip () for sentence in sentences if sentence.strip()]

    
if "__main__" == __name__:
    # 测试关键词提取
    print(to_keywords("小明硕士毕业于中国科学院计算所（Institute of Computing Technology, Chinese Academy of Sciences），后在日本京都大学（Kyoto University）深造。"))
    # 测试断句
    print(sent_tokenize("这是，第一句。这是第二句吗？是的！一个是苹果；一个是桃子。英文的问号是?英文的感叹号是!英文的分号是;英文的句号是."))