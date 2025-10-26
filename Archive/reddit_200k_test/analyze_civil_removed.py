import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 文件路径
POS_FEAT = 'civil_pred_removed_with_features.csv'
CLUST_SUM = 'civil_clusters_summary.csv'
ORIG = 'reddit_200k_test.csv'

print('加载正例（被误标文明样本特征）...')
df_pos = pd.read_csv(POS_FEAT, encoding='utf-8')
print(f'正例数量: {len(df_pos)}')

print('\n簇摘要:')
try:
    cs = pd.read_csv(CLUST_SUM, encoding='utf-8')
    print(cs)
except Exception as e:
    print('无法读取簇摘要:', e)

# 特征统计
feat_cols = ['num_char','num_words','num_exclaim','num_question','uppercase_ratio','url_count','emoji_count','profanity_flag','sentiment']
print('\n正例特征统计:')
print(df_pos[feat_cols].describe().T[['count','mean','std','50%']])

# 构造负例：从原始数据中抽取礼貌候选且不在正例中的样本
print('\n加载原始数据以构造负例...')
df_orig = pd.read_csv(ORIG, encoding='latin1')
texts = df_orig['body'].astype(str)
polite_keywords = ['thank','thanks','please','respect','appreciate','great','good','thank you']
pat = '|'.join([re.escape(k) for k in polite_keywords])
mask_polite = texts.str.lower().str.contains(pat, na=False)

# 移除那些已经在正例中的文本（依据文本去重）
pos_texts = set(df_pos['body'].astype(str).str.strip().tolist())

candidate_texts = df_orig[mask_polite].copy()
candidate_texts['body_stripped'] = candidate_texts['body'].astype(str).str.strip()
neg_candidates = candidate_texts[~candidate_texts['body_stripped'].isin(pos_texts)]
print(f'礼貌候选总数: {len(candidate_texts)}, 去掉正例后可用负例数: {len(neg_candidates)}')

n_pos = len(df_pos)
if len(neg_candidates) >= n_pos:
    df_neg = neg_candidates.sample(n=n_pos, random_state=42).copy()
else:
    df_neg = neg_candidates.copy()

# 提取与正例相同的特征函数（复用简单版本）
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
except Exception:
    analyzer = None

def extract_features_simple(text):
    t = str(text)
    num_char = len(t)
    num_words = len(t.split())
    num_exclaim = t.count('!')
    num_question = t.count('?')
    uppercase_ratio = sum(1 for ch in t if ch.isupper()) / (num_char + 1)
    url_count = len(re.findall(r'http[s]?://', t))
    emoji_count = len(re.findall(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF]', t))
    profanity_list = ['idiot','moron','stupid','dumb']
    profanity_flag = int(any(w in t.lower() for w in profanity_list))
    sentiment = analyzer.polarity_scores(t)['compound'] if analyzer is not None else 0.0
    return pd.Series({
        'num_char': num_char,
        'num_words': num_words,
        'num_exclaim': num_exclaim,
        'num_question': num_question,
        'uppercase_ratio': uppercase_ratio,
        'url_count': url_count,
        'emoji_count': emoji_count,
        'profanity_flag': profanity_flag,
        'sentiment': sentiment
    })

print('\n为负例提取特征...')
df_neg_feats = df_neg['body'].astype(str).apply(extract_features_simple)
df_neg_full = pd.concat([df_neg.reset_index(drop=True), df_neg_feats.reset_index(drop=True)], axis=1)

# 统一列并训练可解释模型
print('\n构建训练集...')
X_pos = df_pos[feat_cols].fillna(0)
X_neg = df_neg_full[feat_cols].fillna(0)
X = pd.concat([X_pos, X_neg], axis=0).reset_index(drop=True)
y = np.array([1]*len(X_pos) + [0]*len(X_neg))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

print('\n可解释模型性能（在负/正样本上）:')
print(classification_report(y_test, clf.predict(X_test)))

# 特征重要性
coef = clf.coef_[0]
feat_imp = sorted(list(zip(feat_cols, coef)), key=lambda x: abs(x[1]), reverse=True)
print('\n特征及系数(按绝对值排序):')
for f,c in feat_imp:
    print(f, c)

# 为每个簇输出典型示例（若存在簇文件）
print('\n每个簇的示例:')
try:
    df_clusters = pd.read_csv('civil_pred_removed_clusters.csv', encoding='utf-8')
    for cluster in sorted(df_clusters['cluster'].unique()):
        print('\nCluster', cluster)
        ex = df_clusters[df_clusters['cluster']==cluster]['body'].astype(str).head(5).tolist()
        for e in ex:
            print('-', e[:200].replace('\n',' '))
except Exception as e:
    print('无法读取带簇文件:', e)

# 保存模型系数与统计为报告
report = {
    'n_pos': len(X_pos),
    'n_neg': len(X_neg),
    'feature_importance': feat_imp
}
pd.Series(report).to_json('civil_analysis_report.json')
print('\n分析完成，已保存 civil_analysis_report.json')
print('生成的CSV文件：civil_pred_removed.csv, civil_pred_removed_with_features.csv, civil_pred_removed_clusters.csv, civil_clusters_summary.csv')
print('如需我把部分示例和统计粘贴到这里，请回复：显示摘要或显示簇X示例')

# === 伪文明 embedding 与情绪指标相关性分析 ===
print("\n[Embedding 与情绪指标相关性分析]")
try:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD
    df = pd.read_csv('civil_pred_removed_with_features.csv')
    # embedding 列转为 np.array
    if 'embedding' in df.columns:
        embeddings = np.vstack(df['embedding'].apply(eval).values)
        svd = TruncatedSVD(n_components=2, random_state=42)
        emb_2d = svd.fit_transform(embeddings)
        # 情绪指标
        vader_compound = df['vader_compound'].values if 'vader_compound' in df.columns else None
        vader_pos = df['vader_pos'].values if 'vader_pos' in df.columns else None
        vader_neg = df['vader_neg'].values if 'vader_neg' in df.columns else None
        # 相关性分析
        if vader_compound is not None:
            corr_compound = np.corrcoef(emb_2d[:,0], vader_compound)[0,1]
            print(f'Embedding 第一主成分与 VADER compound 相关性: {corr_compound:.3f}')
        if vader_pos is not None:
            corr_pos = np.corrcoef(emb_2d[:,0], vader_pos)[0,1]
            print(f'Embedding 第一主成分与 VADER pos 相关性: {corr_pos:.3f}')
        if vader_neg is not None:
            corr_neg = np.corrcoef(emb_2d[:,0], vader_neg)[0,1]
            print(f'Embedding 第一主成分与 VADER neg 相关性: {corr_neg:.3f}')
        # 可视化
        if vader_compound is not None:
            plt.figure(figsize=(7,5))
            sc = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=vader_compound, cmap='coolwarm', alpha=0.7)
            plt.colorbar(sc, label='VADER Compound')
            plt.title('伪文明评论 embedding 与情绪分布')
            plt.xlabel('SVD-1')
            plt.ylabel('SVD-2')
            plt.tight_layout()
            plt.show()
        else:
            print('未找到 VADER 情绪指标，无法可视化。')
    else:
        print('未找到 embedding 列，无法分析。')
except Exception as e:
    print('embedding-情绪分析出错:', e)
