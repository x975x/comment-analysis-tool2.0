"""
LDA 分析核心模块
使用 scikit-learn + Plotly，兼容所有 Python 版本，无需编译。
"""
import io
import math
import os
import re
import tempfile

import jieba
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances


# ── 中文字体（matplotlib 用）─────────────────────────────────
def _set_chinese_font():
    import matplotlib.font_manager as fm
    import os
    for root, _, files in os.walk('/usr/share/fonts'):
        for f in files:
            if f.lower().endswith(('.ttf', '.ttc', '.otf')):
                try:
                    fm.fontManager.addfont(os.path.join(root, f))
                except Exception:
                    pass
    available = {f.name for f in fm.fontManager.ttflist}
    for font in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Microsoft YaHei',
                 'SimHei', 'Noto Sans SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']:
        if font in available:
            matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            break
    matplotlib.rcParams['axes.unicode_minus'] = False

_set_chinese_font()


# ── 停用词 ────────────────────────────────────────────────────
_DEFAULT_STOPWORDS = set([
    '的', '了', '是', '在', '和', '有', '我', '也', '不', '就', '都', '说', '这', '那',
    '他', '她', '它', '我们', '你们', '他们', '这个', '那个', '什么', '怎么', '为什么',
])

def load_stopwords(file_obj=None):
    if file_obj is None:
        return _DEFAULT_STOPWORDS.copy()
    try:
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return set(line.strip() for line in content.splitlines() if line.strip())
    except Exception:
        return _DEFAULT_STOPWORDS.copy()


# ── 自定义词典 ────────────────────────────────────────────────
def load_custom_dict(file_obj):
    if file_obj is None:
        return
    tmp = tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False)
    tmp.write(file_obj.read())
    tmp.close()
    try:
        jieba.load_userdict(tmp.name)
    finally:
        os.unlink(tmp.name)


# ── 文本预处理 ────────────────────────────────────────────────
def preprocess_comments(df, text_col, stopwords):
    docs = []
    for comment in df[text_col].dropna():
        if not isinstance(comment, str):
            continue
        text = re.sub(r'[^一-龥a-zA-Z0-9]', ' ', comment)
        words = [
            w.strip() for w in jieba.lcut(text)
            if len(w.strip()) >= 2 and w.strip() not in stopwords
        ]
        if words:
            docs.append(words)
    return docs


# ── 构建词频矩阵 ──────────────────────────────────────────────
def build_corpus(docs, no_below=3, no_above=0.6):
    texts = [' '.join(doc) for doc in docs]
    vectorizer = CountVectorizer(
        min_df=no_below,
        max_df=no_above,
        token_pattern=r'(?u)\b\w\w+\b',
    )
    dtm = vectorizer.fit_transform(texts)
    return vectorizer, dtm


# ── 自动选主题数（Perplexity）────────────────────────────────
def compute_coherence_scores(dtm, vectorizer, docs, start=2, end=8):
    topic_range = list(range(start, end + 1))
    perplexities, model_list = [], []

    for k in topic_range:
        lda = LatentDirichletAllocation(
            n_components=k, random_state=42,
            max_iter=10, learning_method='online',
        )
        lda.fit(dtm)
        model_list.append(lda)
        perplexities.append(lda.perplexity(dtm))

    best_idx = int(np.argmin(perplexities))
    best_k   = topic_range[best_idx]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(topic_range, perplexities, marker='o', color='steelblue', linewidth=2, markersize=8)
    ax.axvline(x=best_k, color='tomato', linestyle='--', alpha=0.8, label=f'Best K = {best_k}')
    ax.set_xlabel('Number of Topics', fontsize=12)
    ax.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax.set_title('Perplexity Score by Number of Topics', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return best_k, perplexities, buf.getvalue(), model_list[best_idx]


# ── 训练 LDA 模型 ─────────────────────────────────────────────
def train_lda_model(dtm, num_topics, passes=20):
    lda = LatentDirichletAllocation(
        n_components=num_topics, random_state=42,
        max_iter=passes, learning_method='online',
    )
    lda.fit(dtm)
    return lda


# ── 主题关键词表 ──────────────────────────────────────────────
def get_topics_df(lda_model, vectorizer, num_words=12):
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for idx, topic_vec in enumerate(lda_model.components_):
        top_idx  = topic_vec.argsort()[:-num_words - 1:-1]
        keywords = [feature_names[i] for i in top_idx]
        rows.append({'主题编号': f'Topic {idx}', '关键词': '，'.join(keywords)})
    return pd.DataFrame(rows)


# ── 交互式 Plotly 可视化（替代 pyLDAvis）────────────────────
_COLORS = [
    '#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B',
    '#EECA3B', '#B279A2', '#FF9DA6', '#9D755D', '#BAB0AC',
]

def get_topic_plotly(lda_model, dtm, vectorizer, num_words=15):
    """
    生成双面板 Plotly 可视化：
    - 左侧：主题气泡图（按主题占比定大小，MDS 定位）
    - 右侧：当前主题关键词横向条形图
    通过上方按钮切换主题，悬停查看精确数值。
    """
    feature_names = vectorizer.get_feature_names_out()
    n_topics      = lda_model.n_components

    # 计算主题占比
    topic_doc   = lda_model.transform(dtm)           # (n_docs, n_topics)
    prevalence  = topic_doc.mean(axis=0)             # (n_topics,)

    # 用余弦距离 + MDS 给主题定位
    from sklearn.preprocessing import normalize
    topic_word_norm = normalize(lda_model.components_, norm='l1')
    dist_matrix     = cosine_distances(topic_word_norm)

    if n_topics >= 3:
        from sklearn.manifold import MDS
        mds       = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
        positions = mds.fit_transform(dist_matrix)
    elif n_topics == 2:
        positions = np.array([[-1.0, 0.0], [1.0, 0.0]])
    else:
        positions = np.array([[0.0, 0.0]])

    # ── 左侧气泡图的 traces ─────────────────────────────────
    bubble_traces = []
    for idx in range(n_topics):
        bubble_traces.append(go.Scatter(
            x=[positions[idx, 0]],
            y=[positions[idx, 1]],
            mode='markers+text',
            name=f'Topic {idx}',
            marker=dict(
                size=max(30, prevalence[idx] * 600),
                color=_COLORS[idx % len(_COLORS)],
                opacity=0.75,
                line=dict(width=2, color='white'),
            ),
            text=[f'Topic {idx}<br>{prevalence[idx]*100:.1f}%'],
            textposition='middle center',
            textfont=dict(size=11, color='white'),
            hovertemplate=(
                f'<b>Topic {idx}</b><br>'
                f'占比: {prevalence[idx]*100:.1f}%<extra></extra>'
            ),
            xaxis='x', yaxis='y',
        ))

    # ── 右侧关键词条形图的 traces ────────────────────────────
    bar_traces = []
    for idx, topic_vec in enumerate(lda_model.components_):
        top_idx      = topic_vec.argsort()[:-num_words - 1:-1]
        top_words    = [feature_names[i] for i in top_idx]
        top_weights  = topic_vec[top_idx]
        top_weights  = top_weights / top_weights.sum()

        bar_traces.append(go.Bar(
            x=top_weights,
            y=top_words,
            orientation='h',
            name=f'Topic {idx} 关键词',
            marker_color=_COLORS[idx % len(_COLORS)],
            hovertemplate='<b>%{y}</b>  相对权重: %{x:.3f} (%{x:.1%})<extra></extra>',
            xaxis='x2', yaxis='y2',
        ))

    # ── 组合所有 traces：气泡 + 关键词（初始只显示 Topic 0）──
    all_traces = bubble_traces + bar_traces
    n = n_topics
    for i, tr in enumerate(all_traces):
        if i < n:                    # 气泡图：全部显示
            tr.visible = True
        else:                        # 关键词图：只显示 Topic 0
            tr.visible = (i - n == 0)

    # ── 主题切换按钮 ─────────────────────────────────────────
    buttons = []
    for idx in range(n_topics):
        vis = [True] * n + [i == idx for i in range(n)]   # 气泡全显 + 对应关键词显示
        buttons.append(dict(
            label=f'Topic {idx}',
            method='update',
            args=[
                {'visible': vis},
                {'annotations': [dict(
                    text=f'<b>Topic {idx}</b>  关键词权重分布',
                    x=0.78, y=1.04, xref='paper', yref='paper',
                    showarrow=False, font=dict(size=14),
                )]}
            ],
        ))

    # ── 布局 ─────────────────────────────────────────────────
    fig = go.Figure(data=all_traces)
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='right',
            buttons=buttons,
            x=0, xanchor='left',
            y=1.08, yanchor='top',
            bgcolor='#f5f5f5',
            bordercolor='#cccccc',
            font=dict(size=12),
            showactive=True,
            active=0,
        )],
        annotations=[dict(
            text='<b>Topic 0</b>  关键词权重分布',
            x=0.78, y=1.04, xref='paper', yref='paper',
            showarrow=False, font=dict(size=14),
        )],
        xaxis=dict(
            domain=[0, 0.40],
            anchor='y',
            showgrid=False, zeroline=False, showticklabels=False,
            title='主题空间（点击按钮切换）',
        ),
        yaxis=dict(
            domain=[0, 1],
            anchor='x',
            showgrid=False, zeroline=False, showticklabels=False,
        ),
        xaxis2=dict(
            domain=[0.50, 1.0],
            anchor='y2',
            title='关键词相对权重（展示词之和=1）',
            showgrid=True, gridcolor='#f0f0f0',
        ),
        yaxis2=dict(
            domain=[0, 1],
            anchor='x2',
            autorange='reversed',
            showgrid=False,
            side='left',
        ),
        showlegend=False,
        height=max(480, num_words * 28 + 100),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=10, r=10, t=70, b=40),
    )

    return fig


# ── 静态 matplotlib 图（供下载）────────────────────────────
def get_topic_visualization(lda_model, vectorizer, num_words=10):
    feature_names = vectorizer.get_feature_names_out()
    n_topics = lda_model.n_components
    n_cols   = min(3, n_topics)
    n_rows   = math.ceil(n_topics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx in range(n_topics):
        row, col    = divmod(idx, n_cols)
        ax          = axes[row, col]
        topic_vec   = lda_model.components_[idx]
        top_idx     = topic_vec.argsort()[:-num_words - 1:-1][::-1]
        top_words   = [feature_names[i] for i in top_idx]
        top_weights = topic_vec[top_idx]
        top_weights = top_weights / top_weights.sum()

        ax.barh(top_words, top_weights,
                color=_COLORS[idx % len(_COLORS)], edgecolor='white', linewidth=0.5)
        ax.set_title(f'Topic {idx}', fontsize=13, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel('Normalized Weight', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for idx in range(n_topics, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.suptitle('Topic Keyword Distribution', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()
