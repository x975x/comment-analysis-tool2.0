"""
评论文本智能分析工具 2.0 — Streamlit 主程序
运行方式：streamlit run app.py
"""
import streamlit as st
import os as _os
import tempfile
import json as _json
import pandas as pd

from lda_core import (
    load_stopwords as lda_load_stopwords,
    load_custom_dict as lda_load_dict,
    preprocess_comments,
    build_corpus,
    compute_coherence_scores,
    train_lda_model,
    get_topics_df,
    get_topic_plotly,
    get_topic_visualization,
)
from cluster_core import (
    load_stopwords as cluster_load_stopwords,
    load_custom_dict as cluster_load_dict,
    load_topic_keywords,
    run_analysis,
)

# ── 页面配置 ─────────────────────────────────────────────────
st.set_page_config(
    page_title="评论文本智能分析工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局样式 ─────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2rem; font-weight: 700;
    color: #1a73e8; text-align: center;
    padding: 0.5rem 0 1.2rem 0;
}
.stButton > button { width: 100%; border-radius: 8px; font-size: 1rem; }
div[data-testid="metric-container"] {
    background: #f0f4ff; border-radius: 10px; padding: 0.6rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── 编辑器内容磁盘持久化 ─────────────────────────────────────
# 将停用词 / 词典内容保存到本地 JSON，刷新或重连后自动恢复，无需重新输入。
_STATE_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.editor_state.json')

def _load_editor_persist() -> dict:
    try:
        with open(_STATE_FILE, 'r', encoding='utf-8') as _f:
            return _json.load(_f)
    except Exception:
        return {}

def _save_editor_persist(stop_text: str, dict_text: str) -> None:
    try:
        with open(_STATE_FILE, 'w', encoding='utf-8') as _f:
            _json.dump(
                {'lda_stop_text': stop_text, 'lda_dict_text': dict_text},
                _f, ensure_ascii=False, indent=2,
            )
    except Exception:
        pass  # 磁盘不可写时静默忽略，不影响主功能

_persisted_editor = _load_editor_persist()

# ── 示例文件预加载 ────────────────────────────────────────────
def _load_sample(filename):
    path = _os.path.join(_os.path.dirname(__file__), 'samples', filename)
    try:
        with open(path, 'rb') as f:
            return f.read()
    except Exception:
        return None

_sample_csv  = _load_sample('user_comments.csv')
_sample_kw   = _load_sample('topic_keywords.txt')
_sample_dict = _load_sample('my_custom_dict.txt')
_sample_stop = _load_sample('stopwords.txt')

# ── Session state 初始化 ─────────────────────────────────────
_ss_defaults = {
    'lda_results':   None,
    'kw_results':    None,
    # 缓存 CSV 数据（避免重新分析时重传）
    'lda_df':        None,
    'lda_text_col':  'comment',
    # 可编辑的停用词 / 词典文本：优先从磁盘恢复，否则用示例文件内容
    'lda_stop_text': _persisted_editor.get(
        'lda_stop_text', _sample_stop.decode('utf-8') if _sample_stop else ''
    ),
    'lda_dict_text': _persisted_editor.get('lda_dict_text', ''),
    # 词典编辑器触发重分析的标志（跨 rerun 传递，不用局部变量）
    'lda_autorun':   False,
}
for k, v in _ss_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── 辅助函数 ─────────────────────────────────────────────────
def _stopwords_from_text(text: str) -> set:
    """从可编辑文本框内容解析停用词集合（统一转小写，不区分大小写）；空文本则返回内置默认值。"""
    if text.strip():
        return set(line.strip().lower() for line in text.splitlines() if line.strip())
    return lda_load_stopwords(None)

def _load_dict_from_text(text: str):
    """将词典文本写入临时文件后加载到 jieba。"""
    if not text.strip():
        return
    import jieba
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False, encoding='utf-8'
    )
    tmp.write(text)
    tmp.close()
    try:
        jieba.load_userdict(tmp.name)
    finally:
        _os.unlink(tmp.name)

def _nav_link(href: str, label: str) -> str:
    """生成带蓝色圆角箭头的可点击导航链接 HTML。
    使用 onclick+return false 阻止 URL hash 变化（避免 Streamlit Cloud 路由拦截导致 session 丢失），
    改由 scrollIntoView 完成平滑滚动。"""
    aid = href.lstrip('#')
    onclick = (
        f"var e=document.getElementById('{aid}');"
        f"if(e){{e.scrollIntoView({{behavior:'smooth',block:'start'}});}}"
        f"return false;"
    )
    return (
        f'<a href="{href}" onclick="{onclick}" '
        f'style="display:flex;align-items:center;text-decoration:none;'
        f'color:#262730;padding:5px 2px;font-size:0.95rem;">'
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'background:#1a73e8;color:white;border-radius:6px;'
        f'width:22px;height:22px;font-size:16px;margin-right:8px;'
        f'flex-shrink:0;font-weight:bold;line-height:1;">›</span>'
        f'{label}</a>'
    )

def _anchor(aid: str) -> None:
    """在当前位置插入一个 HTML 锚点，供侧边栏导航跳转使用。
    负偏移 80px 确保标题不被 Streamlit 顶部工具栏遮挡。"""
    st.markdown(
        f'<a id="{aid}" style="display:block;position:relative;'
        f'top:-80px;visibility:hidden;"></a>',
        unsafe_allow_html=True,
    )

# ── 页面标题 ─────────────────────────────────────────────────
st.markdown('<div class="main-title">📊 评论文本智能分析工具</div>',
            unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  侧边栏
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ 选择分析模式")
    mode = st.radio(
        "模式",
        ["📡 模式一：LDA 探索性分析", "🎯 模式二：关键词定向分析"],
        label_visibility="collapsed",
    )
    st.divider()
    if "LDA" in mode:
        st.info(
            "**📡 LDA 探索性分析**\n\n"
            "适合新产品 / 陌生领域。\n"
            "不需要预设主题，算法自动从评论中发现潜在话题，并生成交互式可视化图。"
        )
    else:
        st.info(
            "**🎯 关键词定向分析**\n\n"
            "适合已知产品关键词。选择主题（如音质、续航、操作）"
            "统计每个主题的提及次数与权重。"
        )
    st.divider()
    st.caption("上传新文件后点击「开始分析」即可刷新结果。")

    # ── 页面导航目录 ─────────────────────────────────────────
    st.divider()
    st.markdown("#### 📌 页面导航")

    has_lda = st.session_state['lda_results'] is not None
    has_kw  = st.session_state['kw_results']  is not None

    if "LDA" in mode:
        nav_parts = [
            _nav_link("#lda-upload", "上传文件"),
            _nav_link("#lda-params", "参数设置"),
            _nav_link("#lda-run",    "开始分析"),
        ]
        if has_lda:
            nav_parts += [
                _nav_link("#lda-editor",   "调整词典 & 重分析"),
                _nav_link("#lda-results",  "指标概览"),
                _nav_link("#lda-topics",   "主题关键词表"),
                _nav_link("#lda-viz",      "LDA 交互可视化"),
                _nav_link("#lda-download", "下载结果"),
            ]
        else:
            nav_parts.append(
                '<div style="color:#aaa;font-size:0.82rem;'
                'padding:4px 2px 0 30px;">完成分析后显示更多…</div>'
            )
        st.markdown(''.join(nav_parts), unsafe_allow_html=True)
    else:
        nav_parts = [
            _nav_link("#kw-upload", "上传文件"),
            _nav_link("#kw-params", "参数设置"),
            _nav_link("#kw-run",    "开始分析"),
        ]
        if has_kw:
            nav_parts += [
                _nav_link("#kw-results",  "指标概览"),
                _nav_link("#kw-stats",    "主题统计表"),
                _nav_link("#kw-charts",   "可视化图表"),
                _nav_link("#kw-examples", "示例评论"),
                _nav_link("#kw-download", "下载结果"),
            ]
        else:
            nav_parts.append(
                '<div style="color:#aaa;font-size:0.82rem;'
                'padding:4px 2px 0 30px;">完成分析后显示更多…</div>'
            )
        st.markdown(''.join(nav_parts), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  模式一：LDA 探索性分析
# ════════════════════════════════════════════════════════════
if "LDA" in mode:
    st.markdown("## 📡 模式一：LDA 探索性主题分析")

    _anchor("lda-upload")
    col_up, col_par = st.columns([1.05, 1], gap="large")

    with col_up:
        st.markdown("### 📁 上传文件")
        csv_file  = st.file_uploader(
            "评论数据 *（CSV，必须包含评论列）", type=["csv"], key="lda_csv"
        )
        text_col  = st.text_input("评论所在列名", value="comment", key="lda_col")
        dict_file = st.file_uploader(
            "自定义词典（txt，可选）", type=["txt"], key="lda_dict"
        )
        stop_file = st.file_uploader(
            "停用词文件（txt，可选）", type=["txt"], key="lda_stop"
        )

        st.markdown("#### 📥 下载示例文件")
        dl1, dl2, dl3 = st.columns(3)
        if _sample_csv:
            dl1.download_button("💬 评论示例 CSV", data=_sample_csv,
                                file_name="user_comments.csv", mime="text/csv",
                                use_container_width=True, key="lda_dl_csv")
        if _sample_dict:
            dl2.download_button("📖 自定义词典", data=_sample_dict,
                                file_name="my_custom_dict.txt", mime="text/plain",
                                use_container_width=True, key="lda_dl_dict")
        if _sample_stop:
            dl3.download_button("🚫 停用词文件", data=_sample_stop,
                                file_name="stopwords.txt", mime="text/plain",
                                use_container_width=True, key="lda_dl_stop")

    with col_par:
        _anchor("lda-params")
        st.markdown("### 🔧 参数设置")
        auto_k = st.checkbox(
            "🔍 自动选择最佳主题数（Coherence Score，较慢）", value=False
        )
        if auto_k:
            krange = st.slider("主题数搜索范围", min_value=2, max_value=15, value=(2, 8))
        else:
            num_topics = st.slider("主题数量", 2, 20, 5)

        passes   = st.slider("训练轮次 (passes)", 5, 50, 20,
                             help="轮次越高结果越稳定，但训练越慢")
        no_below = st.slider("词频下限 (no_below)", 1, 20, 3,
                             help="词语至少出现在 N 篇文档中才保留")
        no_above = st.slider("词频上限 (no_above)", 0.3, 0.95, 0.6, 0.05,
                             help="词语出现在超过 X% 的文档中则过滤")
        num_words = st.slider("每主题展示关键词数", 5, 20, 12)

    st.markdown("")
    _anchor("lda-run")
    run_lda = st.button("🚀 开始 LDA 分析", type="primary", use_container_width=True)

    # 在渲染编辑器之前读取并立即重置 autorun 标志，避免重复触发
    _autorun = st.session_state['lda_autorun']
    if _autorun:
        st.session_state['lda_autorun'] = False

    # ── 在线调整停用词 / 词典 ────────────────────────────────
    # st.form 防止打字触发 rerun（消除灰屏）。
    # 提交后只做"保存 + 设置 flag + st.rerun()"这一次轻量 rerun，
    # 真正的分析在 flag=True 的下一次 rerun 中执行，避免长分析占用连接。
    _anchor("lda-editor")
    if st.session_state['lda_results'] is not None:
        with st.expander("✏️ 调整停用词 / 自定义词典（可边看结果边修改，无需重传 CSV）"):
            st.caption("直接在下方文本框中增删词条，完成后点击「保存并重新分析」。")
            with st.form("lda_editor_form"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown("**🚫 停用词**（每行一个词）")
                    st.caption(
                        "💡 停用词匹配不区分大小写：填写 `ok` 可同时过滤 `OK`、`Ok`、`ok`。  \n"
                        "💾 点击「保存并重新分析」后内容自动保存到服务器，刷新页面后仅需重新上传csv文件。"
                    )
                    new_stop = st.text_area(
                        label="停用词编辑区",
                        value=st.session_state['lda_stop_text'],
                        height=260,
                        key="lda_stop_editor",
                        label_visibility="collapsed",
                        help="列在此处的词在分词后会被过滤掉，不参与主题建模",
                    )
                with ec2:
                    st.markdown("**📖 自定义词典**（每行一个词）")
                    new_dict = st.text_area(
                        label="词典编辑区",
                        value=st.session_state['lda_dict_text'],
                        height=260,
                        key="lda_dict_editor",
                        label_visibility="collapsed",
                        help="列在此处的词会被 jieba 识别为整体，不会被拆分",
                    )
                save_submitted = st.form_submit_button(
                    "💾 保存并重新分析",
                    type="primary",
                    use_container_width=True,
                )

            # 表单提交：保存编辑内容 → 持久化到磁盘 → 设置 flag → 轻量 rerun
            # 下一次 rerun 时 _autorun=True，才真正跑分析
            if save_submitted:
                st.session_state['lda_stop_text'] = new_stop
                st.session_state['lda_dict_text'] = new_dict
                _save_editor_persist(new_stop, new_dict)   # 刷新/重连后可恢复
                st.session_state['lda_autorun']   = True
                st.rerun()

            # 下载按钮在表单外，读取已保存的 session_state 内容
            dc1, dc2 = st.columns(2)
            dc1.download_button(
                "⬇️ 下载当前停用词文件",
                data=st.session_state['lda_stop_text'].encode('utf-8'),
                file_name="stopwords_edited.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_stop_edited",
            )
            dc2.download_button(
                "⬇️ 下载当前自定义词典",
                data=st.session_state['lda_dict_text'].encode('utf-8'),
                file_name="custom_dict_edited.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_dict_edited",
            )

    # ── 运行分析（首次点击「开始分析」或词典编辑器触发的 autorun）────────
    if run_lda or _autorun:

        # ── 确定 DataFrame 来源 ──────────────────────────────
        if run_lda:
            if csv_file is not None:
                # 有新上传的 CSV：读取并缓存
                df = pd.read_csv(csv_file, encoding='utf-8')
                st.session_state['lda_df']       = df
                st.session_state['lda_text_col'] = text_col
            elif st.session_state['lda_df'] is not None:
                # CSV 控件为空（重连/rerun 后丢失对象）但有缓存数据：静默复用
                df       = st.session_state['lda_df']
                text_col = st.session_state['lda_text_col']
            else:
                st.error("❌ 请先上传评论 CSV 文件！")
                st.stop()
            # 若本次上传了新的停用词 / 词典文件，读取并覆盖 session_state，同步持久化
            _file_changed = False
            if stop_file is not None:
                raw = stop_file.read()
                st.session_state['lda_stop_text'] = (
                    raw.decode('utf-8') if isinstance(raw, bytes) else raw
                )
                _file_changed = True
            if dict_file is not None:
                raw = dict_file.read()
                st.session_state['lda_dict_text'] = (
                    raw.decode('utf-8') if isinstance(raw, bytes) else raw
                )
                _file_changed = True
            if _file_changed:
                _save_editor_persist(
                    st.session_state['lda_stop_text'],
                    st.session_state['lda_dict_text'],
                )
        else:   # _autorun：CSV 来自缓存，无需重传
            if st.session_state['lda_df'] is None:
                st.error("❌ 缓存数据丢失，请重新上传 CSV 文件后再分析。")
                st.stop()
            df       = st.session_state['lda_df']
            text_col = st.session_state['lda_text_col']

        # ── 执行分析 ─────────────────────────────────────────
        _analysis_ok = False
        with st.status("⏳ 正在分析，请稍候…", expanded=True) as status:
            try:
                if text_col not in df.columns:
                    status.update(label="❌ 列名错误", state="error")
                    st.error(
                        f"找不到列 **'{text_col}'**，"
                        f"CSV 中的列为：{list(df.columns)}"
                    )
                    st.stop()
                st.write(f"✅ 共读取 **{len(df)}** 条评论")

                st.write("📚 加载停用词 / 自定义词典…")
                stopwords = _stopwords_from_text(st.session_state['lda_stop_text'])
                _load_dict_from_text(st.session_state['lda_dict_text'])

                st.write("✂️ 文本分词与预处理…")
                docs = preprocess_comments(df, text_col, stopwords)
                if len(docs) < 10:
                    status.update(label="❌ 文档数不足", state="error")
                    st.error(
                        f"有效文档数仅 {len(docs)} 篇，"
                        "太少无法分析，请检查数据或停用词。"
                    )
                    st.stop()
                if len(docs) < 50:
                    st.warning(f"⚠️ 有效文档数较少（{len(docs)} 篇），LDA 效果可能不佳。")
                else:
                    st.write(f"✅ 有效文档：**{len(docs)}** 篇")

                st.write("📦 构建词频矩阵…")
                vectorizer, dtm = build_corpus(docs, no_below, no_above)
                if dtm.shape[1] == 0:
                    status.update(label="❌ 词典为空", state="error")
                    st.error("过滤后词典为空，请降低「词频下限」或提高「词频上限」。")
                    st.stop()
                st.write(f"✅ 词典大小：**{dtm.shape[1]}** 个词")

                chart_bytes = None
                if auto_k:
                    st.write(
                        f"🔍 搜索最佳主题数（{krange[0]} ~ {krange[1]}），"
                        "耗时约数分钟…"
                    )
                    best_k, scores, chart_bytes, lda_model = compute_coherence_scores(
                        dtm, vectorizer, docs, krange[0], krange[1]
                    )
                    final_k = best_k
                    st.write(f"✅ 最佳主题数：**{best_k}**")
                else:
                    st.write(
                        f"🧠 训练 LDA 模型（{num_topics} 个主题，迭代={passes}）…"
                    )
                    lda_model = train_lda_model(dtm, num_topics, passes)
                    final_k   = num_topics
                    st.write("✅ 模型训练完成")

                st.write("🎨 生成交互式主题可视化…")
                topics_df       = get_topics_df(lda_model, vectorizer, num_words)
                topic_fig       = get_topic_plotly(lda_model, dtm, vectorizer, num_words)
                topic_viz_bytes = get_topic_visualization(lda_model, vectorizer, num_words)

                st.session_state['lda_results'] = {
                    'topics_df':       topics_df,
                    'topic_fig':       topic_fig,
                    'topic_viz_bytes': topic_viz_bytes,
                    'chart_bytes':     chart_bytes,
                    'final_k':         final_k,
                    'num_docs':        len(docs),
                    'dict_size':       dtm.shape[1],
                }
                status.update(label="✅ 分析完成！", state="complete", expanded=False)
                _analysis_ok = True

            except Exception as e:
                status.update(label="❌ 分析出错", state="error")
                st.error(f"错误信息：{e}")
                st.exception(e)

        # 首次分析（run_lda）后重跑脚本，使侧边栏目录和词典编辑器立即更新。
        # editor_submitted 路径不需要 rerun：结果在同一次脚本执行中即可渲染，
        # 强制 rerun 反而会导致灰色闪烁、连接中断等问题。
        if _analysis_ok and run_lda:
            st.rerun()

    # ── 展示结果 ─────────────────────────────────────────────
    if st.session_state['lda_results']:
        res = st.session_state['lda_results']
        st.divider()
        st.markdown("## 📊 分析结果")

        # ── 指标卡 ──────────────────────────────────────────
        _anchor("lda-results")
        m1, m2, m3 = st.columns(3)
        m1.metric("📄 有效文档数", res['num_docs'])
        m2.metric("📖 词典大小",   res['dict_size'])
        m3.metric("🏷️ 主题数量",   res['final_k'])

        # ── Perplexity 折线图（仅 auto_k 时有）──────────────
        if res['chart_bytes']:
            st.markdown("### 📈 Perplexity 曲线（越低越好）")
            st.image(res['chart_bytes'], use_column_width=True)

        # ── 主题关键词表 ─────────────────────────────────────
        _anchor("lda-topics")
        st.markdown(f"### 🏷️ 主题关键词（共 {res['final_k']} 个主题）")
        st.dataframe(res['topics_df'], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ 下载主题关键词表（CSV）",
            data=res['topics_df'].to_csv(
                index=False, encoding='utf-8-sig'
            ).encode('utf-8-sig'),
            file_name="lda_topics.csv",
            mime="text/csv",
        )

        # ── 交互式 Plotly 可视化 ─────────────────────────────
        _anchor("lda-viz")
        st.markdown("### 🎨 LDA 主题交互式可视化")
        st.caption("💡 点击上方按钮切换主题，右侧条形图实时更新。悬停可查看精确权重值。")
        st.info(
            "**📖 如何判断模型质量**\n\n"
            "**① 气泡分布**：气泡之间距离越远，说明主题区分度越好；"
            "大面积重叠则说明主题相似，建议减少主题数。\n\n"
            "**② 气泡大小**：各主题占比相对均匀为佳（最大/最小不超过3~5倍）；"
            "若一个主题占70%以上，说明主题数设置过多或数据不够丰富。\n\n"
            "**③ 关键词权重**：右侧条形图中排名靠前的词应该具有业务意义"
            "（如「音质」「降噪」），且不同主题之间的高权重词重叠应尽量少。"
        )
        st.plotly_chart(res['topic_fig'], use_container_width=True)

        # ── 静态图下载 ───────────────────────────────────────
        _anchor("lda-download")
        with st.expander("⬇️ 下载静态版本（PNG）"):
            st.image(res['topic_viz_bytes'], use_column_width=True)
            st.download_button(
                "⬇️ 下载主题可视化图表（PNG）",
                data=res['topic_viz_bytes'],
                file_name="lda_topic_visualization.png",
                mime="image/png",
            )


# ════════════════════════════════════════════════════════════
#  模式二：关键词定向分析
# ════════════════════════════════════════════════════════════
else:
    st.markdown("## 🎯 模式二：关键词定向主题分析")

    _anchor("kw-upload")
    col_up, col_par = st.columns([1.05, 1], gap="large")

    with col_up:
        st.markdown("### 📁 上传文件")
        csv_file   = st.file_uploader(
            "评论数据 *（CSV，必须包含评论列）", type=["csv"], key="kw_csv"
        )
        text_col   = st.text_input("评论所在列名", value="comment", key="kw_col")
        topic_file = st.file_uploader(
            "主题关键词文件 *（txt，格式见右侧说明）", type=["txt"], key="kw_topic"
        )
        dict_file  = st.file_uploader(
            "自定义词典（txt，可选）", type=["txt"], key="kw_dict"
        )
        stop_file  = st.file_uploader(
            "停用词文件（txt，可选）", type=["txt"], key="kw_stop"
        )

        st.markdown("#### 📥 下载示例文件")
        dl1, dl2, dl3, dl4 = st.columns(4)
        if _sample_csv:
            dl1.download_button("💬 评论 CSV", data=_sample_csv,
                                file_name="user_comments.csv", mime="text/csv",
                                use_container_width=True, key="kw_dl_csv")
        if _sample_kw:
            dl2.download_button("🏷️ 关键词文件", data=_sample_kw,
                                file_name="topic_keywords.txt", mime="text/plain",
                                use_container_width=True, key="kw_dl_kw")
        if _sample_dict:
            dl3.download_button("📖 自定义词典", data=_sample_dict,
                                file_name="my_custom_dict.txt", mime="text/plain",
                                use_container_width=True, key="kw_dl_dict")
        if _sample_stop:
            dl4.download_button("🚫 停用词文件", data=_sample_stop,
                                file_name="stopwords.txt", mime="text/plain",
                                use_container_width=True, key="kw_dl_stop")

    with col_par:
        _anchor("kw-params")
        st.markdown("### 🔧 参数设置")
        min_score = st.slider(
            "命中阈值 (min_score)", 1, 5, 1,
            help="一条评论命中该主题至少 N 个关键词才计入此主题",
        )
        st.markdown("#### 📄 主题关键词文件格式")
        st.code(
            "音质 : 音质, 声音, 低音, 高音, 音色\n"
            "降噪 : 降噪, 噪音, 静音, ANC\n"
            "续航 : 续航, 电量, 充电, 电池\n"
            "舒适度 : 舒适, 佩戴, 耳压, 重量\n"
            "连接稳定性 : 蓝牙, 配对, 连接, 延迟, 断连",
            language=None,
        )
        st.caption("每行一个主题，冒号左边是主题名，右边是逗号分隔的关键词。")

    st.markdown("")
    _anchor("kw-run")
    run_kw = st.button("🚀 开始关键词分析", type="primary", use_container_width=True)

    # ── 运行分析 ────────────────────────────────────────────
    if run_kw:
        if csv_file is None:
            st.error("❌ 请先上传评论 CSV 文件！")
        elif topic_file is None:
            st.error("❌ 请上传主题关键词文件！")
        else:
            with st.status("⏳ 正在分析，请稍候…", expanded=True) as status:
                try:
                    st.write("📂 读取数据…")
                    df = pd.read_csv(csv_file, encoding='utf-8')
                    if text_col not in df.columns:
                        status.update(label="❌ 列名错误", state="error")
                        st.error(
                            f"找不到列 **'{text_col}'**，"
                            f"CSV 中的列为：{list(df.columns)}"
                        )
                        st.stop()
                    st.write(f"✅ 共读取 **{len(df)}** 条评论")

                    st.write("📚 加载词典与停用词…")
                    stopwords = cluster_load_stopwords(stop_file)
                    if dict_file:
                        cluster_load_dict(dict_file)
                    topics_dict = load_topic_keywords(topic_file)
                    if not topics_dict:
                        status.update(label="❌ 主题文件解析失败", state="error")
                        st.error(
                            "主题关键词文件解析结果为空，"
                            "请检查格式（主题名 : 词1, 词2）。"
                        )
                        st.stop()
                    st.write(
                        f"✅ 已加载 **{len(topics_dict)}** 个主题："
                        f"{list(topics_dict.keys())}"
                    )

                    st.write("🔍 对每条评论进行多标签分类…")
                    result_df, stats_df, bar_fig, pie_fig, example_dict, summary = \
                        run_analysis(df, text_col, topics_dict, stopwords, min_score)

                    if summary is None:
                        status.update(label="⚠️ 没有命中", state="error")
                        st.error(
                            "没有任何评论命中主题，"
                            "请检查关键词文件或降低命中阈值。"
                        )
                        st.stop()

                    st.session_state['kw_results'] = {
                        'result_df':        result_df,
                        'stats_df':         stats_df,
                        'bar_fig':          bar_fig,
                        'pie_fig':          pie_fig,
                        'example_dict':     example_dict,
                        'summary':          summary,
                        'topics_list':      list(topics_dict.keys()),
                        'stats_df_lookup':  stats_df.set_index('主题'),
                    }
                    status.update(label="✅ 分析完成！", state="complete", expanded=False)

                except Exception as e:
                    status.update(label="❌ 分析出错", state="error")
                    st.error(f"错误信息：{e}")
                    st.exception(e)

    # ── 展示结果 ─────────────────────────────────────────────
    if st.session_state['kw_results']:
        res  = st.session_state['kw_results']
        summ = res['summary']
        lkup = res['stats_df_lookup']

        st.divider()
        st.markdown("## 📊 分析结果")

        # 指标卡
        _anchor("kw-results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📄 总评论数",   summ['total'])
        m2.metric("🔢 总提及次数", summ['total_mentions'])
        m3.metric("❓ 未分类评论", summ['unclassified'])
        m4.metric("📉 未分类率",   f"{summ['unclassified_pct']:.1f}%")

        # 统计表
        _anchor("kw-stats")
        st.markdown("### 📋 主题提及统计")
        st.dataframe(res['stats_df'], use_container_width=True, hide_index=True)

        # 图表
        _anchor("kw-charts")
        st.markdown("### 📈 可视化图表")
        col_bar, col_pie = st.columns(2)
        with col_bar:
            st.markdown("#### 📊 各主题提及次数（条形图）")
            st.plotly_chart(res['bar_fig'], use_container_width=True)
        with col_pie:
            st.markdown("#### 🥧 各主题权重分布（饼图）")
            st.plotly_chart(res['pie_fig'], use_container_width=True)

        # 示例评论
        _anchor("kw-examples")
        st.markdown("### 💬 各主题示例评论")
        for topic in res['topics_list']:
            count    = int(lkup.loc[topic, '提及次数']) if topic in lkup.index else 0
            examples = res['example_dict'].get(topic, [])
            with st.expander(f"【{topic}】— 共提及 {count} 次"):
                if examples:
                    for i, ex in enumerate(examples, 1):
                        st.markdown(f"**{i}.** {ex}")
                else:
                    st.caption("该主题暂无匹配评论")

        # 下载
        _anchor("kw-download")
        st.markdown("### ⬇️ 下载结果")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "📄 下载评论分类结果（CSV）",
                data=res['result_df'].to_csv(
                    index=False, encoding='utf-8-sig'
                ).encode('utf-8-sig'),
                file_name="classified_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                "📊 下载主题统计表格（CSV）",
                data=res['stats_df'].to_csv(
                    index=False, encoding='utf-8-sig'
                ).encode('utf-8-sig'),
                file_name="topic_stats.csv",
                mime="text/csv",
                use_container_width=True,
            )
