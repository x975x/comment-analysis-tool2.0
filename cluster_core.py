"""
关键词多标签聚类核心模块
图表使用 Plotly（浏览器渲染，中文显示无问题）。
"""
import os
import re
import tempfile

import jieba
import pandas as pd
import plotly.graph_objects as go

_DEFAULT_STOPWORDS = set([
    '的', '了', '是', '在', '和', '有', '我', '也', '不', '就', '都', '说', '这', '那',
    '他', '她', '它', '我们', '你们', '他们', '这个', '那个', '什么', '怎么', '为什么',
    '感觉', '觉得', '真的', '很', '太', '非常', '有点', '比较', '还行', '不错', '可以',
    '但是', '因为', '所以', '然后', '这样', '那样', '一下', '一直', '一些', '已经',
    '还是', '所有', '大家', '别人', '东西', '产品', '店家', '卖家', '收到', '发货',
    '物流', '快递', '包装', '使用', '体验', '效果', '外观', '材质', '做工', '质量',
    '品牌', '价格', '性价比', '服务', '客服', '态度', '速度', '快', '慢', '好', '差',
    '一般', '满意', '喜欢', '推荐', '购买', '下单', '退货',
])

_COLORS = [
    '#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B',
    '#EECA3B', '#B279A2', '#FF9DA6', '#9D755D', '#BAB0AC',
]


# ── 停用词 ────────────────────────────────────────────────────
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


# ── 主题关键词文件解析 ────────────────────────────────────────
def load_topic_keywords(file_obj):
    topics = {}
    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode('utf-8')
    for line in content.splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        topic, kw_str = line.split(':', 1)
        topic    = topic.strip()
        keywords = [kw.strip() for kw in kw_str.split(',') if kw.strip()]
        if topic and keywords:
            topics[topic] = keywords
    return topics


# ── 分词 ─────────────────────────────────────────────────────
def _segment(comment, stopwords):
    text  = re.sub(r'[^一-龥a-zA-Z0-9]', ' ', str(comment))
    words = jieba.lcut(text)
    return set(w.strip() for w in words if len(w.strip()) >= 2 and w.strip() not in stopwords)


# ── 主分析函数 ────────────────────────────────────────────────
def run_analysis(df, text_col, topics_dict, stopwords, min_score=1):
    """
    多标签分类并生成 Plotly 图表（中文正常显示）。
    返回: (result_df, stats_df, bar_fig, pie_fig, example_dict, summary)
    summary 为 None 表示无命中。
    """
    total   = len(df)
    records = []

    for _, row in df.iterrows():
        comment = str(row[text_col])
        words   = _segment(comment, stopwords)
        hit     = [
            t for t, kws in topics_dict.items()
            if sum(1 for k in kws if k in words) >= min_score
        ]
        records.append({
            '原始评论': comment,
            '命中主题': '，'.join(hit) if hit else '未分类',
            '_hits':    hit,
        })

    topic_count   = {t: sum(1 for r in records if t in r['_hits']) for t in topics_dict}
    unclassified  = sum(1 for r in records if not r['_hits'])
    total_mentions = sum(topic_count.values())

    if total_mentions == 0:
        return None, None, None, None, None, None

    mention_rate = {t: c / total * 100      for t, c in topic_count.items()}
    weight       = {t: c / total_mentions * 100 for t, c in topic_count.items()}

    # ── 统计表 ───────────────────────────────────────────────
    stats_df = pd.DataFrame([{
        '主题':     t,
        '提及次数': topic_count[t],
        '提及率(%)': round(mention_rate[t], 1),
        '权重(%)':   round(weight[t], 1),
    } for t in topics_dict])

    result_df = pd.DataFrame([{
        '原始评论': r['原始评论'],
        '命中主题': r['命中主题'],
    } for r in records])

    topics_list = sorted(topics_dict.keys(), key=lambda t: topic_count[t], reverse=True)
    counts      = [topic_count[t] for t in topics_list]
    weights     = [weight[t]      for t in topics_list]
    bar_colors  = [_COLORS[i % len(_COLORS)] for i in range(len(topics_list))]

    # ── 条形图（Plotly）──────────────────────────────────────
    bar_fig = go.Figure(go.Bar(
        x=topics_list,
        y=counts,
        marker_color=bar_colors,
        text=counts,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>提及次数: %{y}<extra></extra>',
    ))
    bar_fig.update_layout(
        title=dict(text='各主题被提及次数（多标签，一条评论可计入多个主题）', font=dict(size=14)),
        xaxis=dict(title='主题', tickfont=dict(size=12)),
        yaxis=dict(title='提及次数（评论数）', showgrid=True, gridcolor='#f0f0f0'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(t=60, b=60, l=60, r=20),
    )
    bar_fig.update_xaxes(showgrid=False)

    # ── 饼图（Plotly，缩小尺寸）─────────────────────────────
    pie_fig = go.Figure(go.Pie(
        labels=topics_list,
        values=weights,
        marker=dict(colors=bar_colors),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>权重: %{percent}<extra></extra>',
        hole=0.05,
    ))
    pie_fig.update_layout(
        title=dict(text='各主题权重分布（基于提及次数归一化）', font=dict(size=14)),
        height=400,
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation='v', x=1.02, y=0.5),
    )

    # ── 示例评论 ─────────────────────────────────────────────
    example_dict = {
        t: [r['原始评论'] for r in records if t in r['_hits']][:3]
        for t in topics_dict
    }

    summary = {
        'total':            total,
        'total_mentions':   total_mentions,
        'unclassified':     unclassified,
        'unclassified_pct': unclassified / total * 100,
    }

    return result_df, stats_df, bar_fig, pie_fig, example_dict, summary
