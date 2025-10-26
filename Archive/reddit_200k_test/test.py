import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm import tqdm

# --- 1. 配置参数 ---
# 定义常量，方便修改
DATA_FILE = 'reddit_200k_test.csv'  # Kaggle数据集文件名
TEXT_COLUMN = 'body'                      # 包含评论文本的列
TARGET_COLUMN = 'REMOVED'                   # 目标标签列 (True/False)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'   # 一个轻量且高效的英文Embedding模型
SAMPLE_SIZE = 20000                       # 由于数据集很大，我们先取一个子集进行快速实验
RANDOM_STATE = 42                         # 设置随机种子，保证每次运行结果一致

# --- 2. 加载与预处理数据 ---
print("Step 1: 加载与预处理数据...")

# 加载数据。如果内存充足，可以注释掉 .sample() 那一行来使用全部数据
try:
    df = pd.read_csv(DATA_FILE, encoding='latin1')  # 使用 latin1 编码
    print(f"成功加载数据集，总共 {len(df)} 条评论。")
    print("\n可用的列名：")
    print(df.columns.tolist())
    # 对数据进行采样，以便快速验证流程
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    print(f"已采样 {SAMPLE_SIZE} 条数据用于本次实验。")
except FileNotFoundError:
    print(f"错误：找不到数据集文件 '{DATA_FILE}'。请确保它和你的脚本在同一个目录下。")
    exit()

# 数据清洗
# a. 将目标列从布尔值(True/False)转换为整数(1/0)
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

# b. 移除内容为 "[removed]" 或 "[deleted]" 的无效评论，这些是噪音
df = df[~df[TEXT_COLUMN].isin(['[removed]', '[deleted]'])]

# c. 丢弃文本内容为空的行
df.dropna(subset=[TEXT_COLUMN], inplace=True)

print("数据预处理完成。")
print("数据类别分布情况：")
print(df[TARGET_COLUMN].value_counts(normalize=True))


# --- 3. 生成文本 Embedding ---
print("\nStep 2: 生成文本 Embedding...")
print(f"正在加载 Embedding 模型: {EMBEDDING_MODEL_NAME}")

# 加载预训练的 Sentence Transformer 模型
# 第一次运行时会自动下载模型，可能需要一些时间
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# 获取所有评论文本
sentences = df[TEXT_COLUMN].tolist()

# 使用模型将文本转换为向量（Embedding）
# show_progress_bar=True 可以显示一个漂亮的进度条
print("正在将文本转换为 Embedding 向量，这可能需要几分钟...")
embeddings = model.encode(sentences, show_progress_bar=True)
print("Embedding 生成完毕！")
print(f"我们得到了 {embeddings.shape[0]} 个向量，每个向量的维度是 {embeddings.shape[1]}。")


# --- 4. 准备训练数据 ---
print("\nStep 3: 准备训练数据与测试数据...")

# 定义特征 (X) 和标签 (y)
X = embeddings
y = df[TARGET_COLUMN]

# 将数据划分为训练集和测试集
# test_size=0.2 表示80%的数据用于训练，20%用于测试
# stratify=y 是一个非常重要的参数，特别是在数据不平衡时！
# 它能确保训练集和测试集中的类别比例与原始数据保持一致。
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("数据划分完毕。")


# --- 5. 训练分类器 ---
print("\nStep 4: 训练分类器...")

# 我们选择逻辑回归作为分类器，它速度快，效果好
# class_weight='balanced' 是处理数据不平衡问题的利器！
# 它会自动给样本量少的类别（这里是'removed=1'的评论）更高的权重。
classifier = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE)

# 使用训练数据来训练模型
classifier.fit(X_train, y_train)
print("模型训练完成！")


# --- 6. 评估模型 ---
print("\nStep 5: 评估模型性能...")

# 使用训练好的模型对测试集进行预测
y_pred = classifier.predict(X_test)

# 生成并打印分类报告，包含了精确率、召回率和F1分数
print("分类报告:")
# target_names=['Not Removed (0)', 'Removed (1)'] 可以让报告更易读
print(classification_report(y_test, y_pred, target_names=['Not Removed (0)', 'Removed (1)']))


# ------- 附加分析：抽取被误标的“文明但被预测为 removed”的样本并提取特征 -------
print("\nStep 6: 抽取并分析被误标的文明样本...")

# 1) 在整个样本上计算预测和置信度
try:
    probs = classifier.predict_proba(X)[:, 1]
except Exception:
    # 如果 classifier 不支持 predict_proba，则用决策函数近似（很少发生）
    try:
        probs = classifier.decision_function(X)
        # 归一化到 [0,1]
        probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-9)
    except Exception:
        probs = np.zeros(len(X))

preds = classifier.predict(X)

# 将预测结果附加回样本表
df_sample = df.reset_index(drop=True).copy()
df_sample['pred'] = preds
df_sample['prob_removed'] = probs

# 2) 用简单规则标注文明候选（礼貌关键词），可以用人工标注替换
polite_keywords = ['thank', 'thanks', 'please', 'respect', 'appreciate', 'great', 'good', 'please', 'thank you']
pat = '|'.join([re.escape(k) for k in polite_keywords])
df_sample['is_polite_candidate'] = df_sample[TEXT_COLUMN].astype(str).str.lower().str.contains(pat)

# 3) 筛出文明候选但被模型预测为 removed 的样本并保存
civil_pred_removed = df_sample[(df_sample['is_polite_candidate']) & (df_sample['pred'] == 1)].copy()
civ_count = len(civil_pred_removed)
civil_pred_removed.to_csv('civil_pred_removed.csv', index=False, encoding='utf-8')
print(f"已保存 {civ_count} 条被预测为 removed 的文明候选到 civil_pred_removed.csv")

# 4) 为这些样本提取结构化特征并保存
print("开始为被误标样本提取结构化特征...")
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
except Exception:
    analyzer = None

def extract_features(text):
    t = str(text)
    num_char = len(t)
    num_words = len(t.split())
    num_exclaim = t.count('!')
    num_question = t.count('?')
    uppercase_ratio = sum(1 for ch in t if ch.isupper()) / (num_char + 1)
    url_count = len(re.findall(r'http[s]?://', t))
    emoji_count = len(re.findall(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF]', t))
    profanity_list = ['idiot', 'moron', 'stupid', 'dumb']
    profanity_flag = int(any(w in t.lower() for w in profanity_list))
    sentiment = analyzer.polarity_scores(t)['compound'] if analyzer is not None else 0.0
    return {
        'num_char': num_char,
        'num_words': num_words,
        'num_exclaim': num_exclaim,
        'num_question': num_question,
        'uppercase_ratio': uppercase_ratio,
        'url_count': url_count,
        'emoji_count': emoji_count,
        'profanity_flag': profanity_flag,
        'sentiment': sentiment
    }

if civ_count > 0:
    # 索引重置，保证 embedding 行号一致
    civil_pred_removed_reset = civil_pred_removed.reset_index(drop=True)
    feats = civil_pred_removed_reset[TEXT_COLUMN].astype(str).apply(lambda t: pd.Series(extract_features(t)))
    civil_features = pd.concat([civil_pred_removed_reset, feats], axis=1)
    civil_features['embedding'] = [embeddings[idx].tolist() for idx in civil_pred_removed_reset.index]
    civil_features['embedding'] = civil_features['embedding'].apply(str)
    civil_features.to_csv('civil_pred_removed_with_features.csv', index=False, encoding='utf-8')
    print(f"已保存带特征和 embedding 的被误标样本到 civil_pred_removed_with_features.csv (n={len(civil_features)})")
else:
    print('未找到被误标的文明候选，跳过特征提取。')

# 5) 基于 TF-IDF 的聚类 (仅当样本量足够时)
if civ_count >= 10:
    print('开始 TF-IDF 向量化并进行聚类（kmeans）...')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans

    texts = civil_pred_removed[TEXT_COLUMN].astype(str).tolist()
    tfv = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english', ngram_range=(1,2))
    X_tfidf = tfv.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(50, X_tfidf.shape[1]-1), random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    k = min(4, max(2, civ_count // 10))
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_reduced)
    civil_pred_removed['cluster'] = labels
    civil_pred_removed.to_csv('civil_pred_removed_clusters.csv', index=False, encoding='utf-8')

    # 为每个簇保存 top terms
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tfv.get_feature_names_out()
    clusters_summary = []
    for i in range(k):
        top_terms = [terms[ind] for ind in order_centroids[i, :10] if ind < len(terms)]
        size = int((labels == i).sum())
        clusters_summary.append({'cluster': i, 'size': size, 'top_terms': ' '.join(top_terms)})
    pd.DataFrame(clusters_summary).to_csv('civil_clusters_summary.csv', index=False, encoding='utf-8')
    print('已保存聚类结果 civil_pred_removed_clusters.csv 和簇摘要 civil_clusters_summary.csv')
else:
    print('样本量不足，跳过聚类步骤。')

print('\n下一步建议：查看 civil_pred_removed_with_features.csv 的前 200 条样本，人工标注原因（low-quality/snarky/policy/etc），然后我们继续训练可解释模型。')

# 继续原有的 Step 6：分析示例组（如果需要）
print("\nStep 7: 分析文明与不文明示例组（保留原有示例分析）...")

# 准备更多样的文明与不文明评论示例
civil_comments = [
    "This is a fantastic explanation, thank you for sharing your knowledge!",  # 感谢
    "While I disagree, I respect your perspective and here's why...",         # 理性争辩
    "Great point! Could you please elaborate more on that?",                  # 建设性提问
    "The evidence suggests otherwise, but I'm open to discussion.",           # 理性反驳
    "Here are some sources that might help clarify this topic..."            # 提供资源
]

uncivil_comments = [
    "You are an idiot, you don't know what you are talking about.",          # 人身攻击
    "This is totally stupid and you should feel bad.",                        # 侮辱性言论
    "Go back to school you moron!",                                          # 贬低性言论
    "Whatever, this is just garbage posted by trolls.",                      # 无理诽谤
    "lol ur dumb XD"                                                         # 幼稚嘲讽
]

# 生成embeddings
civil_embeddings = model.encode(civil_comments)
uncivil_embeddings = model.encode(uncivil_comments)

# 计算各组内的平均相似度
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_group_similarity(embeddings):
    sim_matrix = cosine_similarity(embeddings)
    # 获取上三角矩阵的平均值（排除对角线的1）
    upper_tri = np.triu(sim_matrix, k=1)
    return np.mean(upper_tri[upper_tri != 0])

civil_internal_sim = calculate_group_similarity(civil_embeddings)
uncivil_internal_sim = calculate_group_similarity(uncivil_embeddings)

print("\n文明评论组内相似度：{:.3f}".format(civil_internal_sim))
print("不文明评论组内相似度：{:.3f}".format(uncivil_internal_sim))

# 计算两组评论之间的距离
cross_sim_matrix = cosine_similarity(civil_embeddings, uncivil_embeddings)
cross_sim = np.mean(cross_sim_matrix)
print("文明与不文明评论之间的平均相似度：{:.3f}".format(cross_sim))

# 使用分类器预测并分析特征重要性
all_comments = civil_comments + uncivil_comments
all_embeddings = model.encode(all_comments)
predictions = classifier.predict(all_embeddings)

print("\n分类结果分析：")
print("文明评论被标记为移除的比例：{:.1%}".format(
    np.mean(predictions[:len(civil_comments)] == 1)
))
print("不文明评论被标记为移除的比例：{:.1%}".format(
    np.mean(predictions[len(civil_comments):] == 1)
))

# 展示具体示例
print("\n典型示例分析：")
for comment, pred in zip(civil_comments + uncivil_comments, predictions):
    result = "可能被移除 (Removed)" if pred == 1 else "可能被保留 (Not Removed)"
    comment_type = "文明评论" if comment in civil_comments else "不文明评论"
    print(f"\n类型：{comment_type}")
    print(f"评论: \"{comment[:50]}...\"")
    print(f"预测结果: {result}")
