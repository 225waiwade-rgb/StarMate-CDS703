import base64
import inspect
import os
import time
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from src.pipeline import StarMatePipeline

load_dotenv()


def find_logo_path() -> str:
    for candidate in ("assets/logo.png", "assets/logo.jpg", "assets/logo.jpeg"):
        if os.path.exists(candidate):
            return candidate
    return ""


LOGO_PATH = find_logo_path()

st.set_page_config(
    page_title="StarMate | AI Course Assistant",
    page_icon=LOGO_PATH if LOGO_PATH else "⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAPER_CSS = """
<style>
:root {
    --paper-bg: #FAF9F6;
    --paper-sidebar: #F3F1EC;
    --paper-card: #FFFFFF;
    --paper-border: #E6E2D8;
    --ink-primary: #1F1E1C;
    --ink-secondary: #6B6860;
    --accent-blue: #2563EB;
    --accent-blue-hover: #1D4ED8;
    --accent-blue-soft: #EFF6FF;
    --font-display: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
    --font-text: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
}

html, body, p, span, label, input, textarea, button, li, div {
    font-family: var(--font-text);
    -webkit-font-smoothing: antialiased;
    letter-spacing: -0.008em;
}

h1, h2, h3, .metric-value {
    font-family: var(--font-display) !important;
    letter-spacing: -0.02em;
}

.stApp {
    background-color: var(--paper-bg);
    color: var(--ink-primary);
}

/* Warm Frosted Glass Header */
header[data-testid="stHeader"] {
    background: rgba(250, 249, 246, 0.82) !important;
    backdrop-filter: saturate(180%) blur(16px);
    -webkit-backdrop-filter: saturate(180%) blur(16px);
    border-bottom: 1px solid rgba(31, 30, 28, 0.06);
}

/* Soft Warm Linen Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--paper-sidebar) !important;
    border-right: 1px solid var(--paper-border);
}

/* Warm Paper Hero Card */
.paper-hero {
    display: flex;
    align-items: center;
    gap: 20px;
    background: linear-gradient(135deg, #FFFFFF 0%, #F6F3EC 100%);
    border: 1px solid var(--paper-border);
    border-radius: 22px;
    padding: 26px 30px;
    margin-bottom: 18px;
    box-shadow: 0 6px 24px rgba(31, 30, 28, 0.04);
}

.paper-eyebrow {
    display: inline-block;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--accent-blue);
    background: var(--accent-blue-soft);
    padding: 4px 10px;
    border-radius: 999px;
    margin-bottom: 10px;
}

.paper-hero h1 {
    margin: 0;
    font-size: 34px;
    line-height: 1.15;
    font-weight: 700;
    color: var(--ink-primary);
}

.paper-hero p {
    margin: 8px 0 0 0;
    font-size: 15px;
    line-height: 1.55;
    color: var(--ink-secondary);
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 14px 0 22px 0;
}

.paper-metric {
    background: var(--paper-card);
    border: 1px solid var(--paper-border);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 2px 10px rgba(31, 30, 28, 0.03);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.paper-metric:hover {
    transform: translateY(-2px);
    border-color: #CBD5E1;
    box-shadow: 0 8px 20px rgba(31, 30, 28, 0.06);
}

.metric-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ink-secondary);
}

.metric-value {
    font-size: 22px;
    font-weight: 700;
    color: var(--ink-primary);
    margin-top: 4px;
}

/* Course Tag Badge */
.course-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    background: var(--accent-blue-soft);
    border: 1px solid #DBEAFE;
    color: var(--accent-blue);
    margin-right: 8px;
}

/* Chat Messages */
div[data-testid="stChatMessage"] {
    background: var(--paper-card);
    border: 1px solid var(--paper-border);
    border-radius: 18px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 2px 10px rgba(31, 30, 28, 0.03);
}

/* Expandable Citation Cards */
div[data-testid="stExpander"] {
    background: #FBF9F5 !important;
    border: 1px solid var(--paper-border) !important;
    border-radius: 14px !important;
}

/* Primary Buttons in Academic Blue */
div.stButton > button[kind="primary"] {
    background-color: var(--accent-blue) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 500;
    transition: background-color 0.2s ease, transform 0.2s ease;
}

div.stButton > button[kind="primary"]:hover {
    background-color: var(--accent-blue-hover) !important;
    transform: translateY(-1px);
}

/* Secondary & Starter Buttons */
div.stButton > button[kind="secondary"] {
    background-color: var(--paper-card) !important;
    color: var(--ink-primary) !important;
    border: 1px solid var(--paper-border) !important;
    border-radius: 16px !important;
    text-align: left;
    box-shadow: 0 2px 8px rgba(31, 30, 28, 0.03);
    transition: border-color 0.2s ease, color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(37, 99, 235, 0.08);
}

/* Warm Paper Login Card */
.login-card {
    max-width: 440px;
    margin: 8vh auto 18px auto;
    background: var(--paper-card);
    border: 1px solid var(--paper-border);
    border-radius: 22px;
    padding: 32px 30px;
    text-align: center;
    box-shadow: 0 12px 36px rgba(31, 30, 28, 0.06);
}

.login-card h2 {
    margin: 12px 0 0 0;
    font-size: 28px;
    font-weight: 700;
    color: var(--ink-primary);
}

.login-card p {
    margin: 10px 0 0 0;
    font-size: 14px;
    color: var(--ink-secondary);
}

@media (max-width: 768px) {
    .metric-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .paper-hero {
        flex-direction: column;
        align-items: flex-start;
    }
    .paper-hero h1 {
        font-size: 26px;
    }
}
</style>
"""

st.markdown(PAPER_CSS, unsafe_allow_html=True)

UI_TEXT = {
    "en": {
        "controls": "StarMate Controls",
        "controls_desc": "Configure retrieval settings or upload your own course PDFs.",
        "course_scope": "Course Scope",
        "all_courses": "All Courses",
        "top_k": "Retrieved Passages (Top-K)",
        "upload_title": "Upload Personal Slides",
        "upload_label": "Add a PDF to this session",
        "course_tag": "Course Tag",
        "default_tag": "My Uploaded Notes",
        "index_btn": "Index Uploaded PDF",
        "indexing": "Extracting and indexing PDF...",
        "uploaded_caption": "Uploaded in this session: ",
        "clear_btn": "Clear Conversation",
        "cleared_toast": "Conversation cleared.",
        "eyebrow": "CDS703 · COURSE RAG STUDY ASSISTANT",
        "subtitle": "Hybrid BM25 + Vector Retrieval with verifiable page-level citations.",
        "info_btn": "System Info",
        "metric_chunks": "Knowledge Chunks",
        "metric_courses": "Indexed Courses",
        "metric_mode": "Retrieval Mode",
        "metric_guard": "Citation Guardrail",
        "guard_val": "Active",
        "starters_title": "### Quick Starter Questions",
        "chat_placeholder": "Ask a question about your course materials...",
        "status_running": "Running StarMate RAG pipeline...",
        "status_step1": "1. Searching indexed passages via **{mode}**...",
        "status_step2": "2. Filtering course scope: **{course}** (Top-{k})...",
        "status_step3": "3. Grounded response generated with {n} cited excerpts.",
        "status_done": "Completed in {t:.1f}s",
        "sources_expander": "View {n} Cited Course Excerpts",
        "dialog_title": "About StarMate Architecture",
        "dialog_body": (
            "**StarMate** is a course-grounded Retrieval-Augmented Generation (RAG) study assistant built for **CDS703**.\n\n"
            "- **Hybrid Retrieval**: Combines lexical **BM25** and dense **all-MiniLM-L6-v2** vector search fused via **Reciprocal Rank Fusion (RRF)**.\n"
            "- **Grounded Citations**: Every answer cites exact source PDF filenames and page numbers, refusing to hallucinate when excerpts are insufficient.\n"
            "- **Dynamic PDF Upload**: Students can upload personal lecture slides for instant in-session Q&A."
        ),
        "dialog_tip": "Tip: Switch between Hybrid, Vector, and BM25 using the segmented control to compare retrieval behavior live.",
    },
    "zh": {
        "controls": "StarMate 控制台",
        "controls_desc": "调整检索参数，或上传你自己的课程 PDF 讲义。",
        "course_scope": "课程筛选范围",
        "all_courses": "全部课程",
        "top_k": "检索片段数量 (Top-K)",
        "upload_title": "上传个人课件",
        "upload_label": "添加 PDF 到本次会话",
        "course_tag": "课程标签",
        "default_tag": "我的上传笔记",
        "index_btn": "解析并加入索引",
        "indexing": "正在解析并索引 PDF...",
        "uploaded_caption": "本次已上传：",
        "clear_btn": "清空对话记录",
        "cleared_toast": "已清空对话记录。",
        "eyebrow": "CDS703 · 课程资料 RAG 学习助手",
        "subtitle": "融合 BM25 与语义向量检索，严格输出可验证的页码级来源引用。",
        "info_btn": "系统说明",
        "metric_chunks": "知识库切片数",
        "metric_courses": "已索引课程数",
        "metric_mode": "当前检索模式",
        "metric_guard": "防幻觉引用",
        "guard_val": "已开启",
        "starters_title": "### 快捷示例提问",
        "chat_placeholder": "输入关于课程资料的问题...",
        "status_running": "正在运行 StarMate RAG 流水线...",
        "status_step1": "1. 正在使用 **{mode}** 检索课程知识库...",
        "status_step2": "2. 课程范围过滤：**{course}**（Top-{k}）...",
        "status_step3": "3. 已基于 {n} 条课程原文生成带引用的回答。",
        "status_done": "已完成（耗时 {t:.1f} 秒）",
        "sources_expander": "查看 {n} 条课程原文引用",
        "dialog_title": "关于 StarMate 系统架构",
        "dialog_body": (
            "**StarMate** 是面向 **CDS703** 课程打造的课程资料 RAG 智能学习助手。\n\n"
            "- **混合检索**：融合关键词检索 **BM25** 与语义向量 **all-MiniLM-L6-v2**，采用 **RRF（倒数排名融合）** 排序。\n"
            "- **防幻觉引用**：回答严格基于课程原文并标注 PDF 文件名与页码；资料不足时拒绝编造。\n"
            "- **动态上传课件**：支持学生在当前会话中即时上传个人 PDF 并提问。"
        ),
        "dialog_tip": "提示：可在上方分段选择器中实时切换 Hybrid、Vector 与 BM25 对比检索效果。",
    },
}


def get_logo_html(width: int = 76) -> str:
    if not LOGO_PATH:
        return ""
    with open(LOGO_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = "jpeg" if LOGO_PATH.lower().endswith((".jpg", ".jpeg")) else "png"
    return (
        f'<img src="data:image/{ext};base64,{b64}" '
        f'style="width:{width}px;height:auto;border-radius:16px;flex-shrink:0;" />'
    )


def get_app_password() -> str:
    env_pw = os.getenv("APP_PASSWORD", "")
    if env_pw:
        return env_pw
    try:
        return st.secrets.get("APP_PASSWORD", "")
    except Exception:
        return ""


def check_access_password() -> None:
    required_pw = get_app_password()
    if not required_pw:
        return
    if st.session_state.get("authenticated", False):
        return

    login_logo = get_logo_html(width=96)
    st.markdown(
        f"""
        <div class="login-card">
            {login_logo}
            <div style="margin-top:12px;">
                <span class="paper-eyebrow">CDS703 · STARMATE</span>
            </div>
            <h2>Welcome to StarMate</h2>
            <p>Course RAG Study Assistant<br/>Enter the access passcode to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        pwd = st.text_input("Access Passcode / 访问口令", type="password", placeholder="Enter passcode...")
        if st.button("Continue / 解锁进入", type="primary", use_container_width=True):
            if pwd == required_pw:
                st.session_state["authenticated"] = True
                st.toast("Access granted.", icon="✓")
                st.rerun()
            else:
                st.error("Incorrect passcode / 口令错误，请重试。")
    st.stop()


check_access_password()


@st.cache_resource(show_spinner="Initializing StarMate Hybrid RAG engine...")
def load_pipeline():
    return StarMatePipeline()


pipeline = load_pipeline()


def get_available_courses(all_label: str):
    chunks = getattr(pipeline.retriever, "chunks", [])
    courses = sorted({c.get("course", "Unknown") for c in chunks if c.get("course")})
    return [all_label] + courses


def run_pipeline_answer(prompt: str, course_filter, mode: str, top_k: int, lang: str):
    sig = inspect.signature(pipeline.answer)
    kwargs = {}
    if "course_filter" in sig.parameters:
        kwargs["course_filter"] = course_filter
    if "mode" in sig.parameters:
        kwargs["mode"] = mode
    if "top_k" in sig.parameters:
        kwargs["top_k"] = top_k
    if "language" in sig.parameters:
        kwargs["language"] = lang

    res = pipeline.answer(prompt, **kwargs)
    if isinstance(res, dict):
        answer = res.get("answer", "")
        sources = res.get("sources") or res.get("chunks") or res.get("retrieved") or []
        return answer, sources
    if isinstance(res, tuple) and len(res) >= 2:
        return res[0], res[1]
    return str(res), []


def add_uploaded_pdf(uploaded_file, course_tag: str) -> int:
    retriever = pipeline.retriever
    for method_name in ("add_uploaded_pdf", "add_pdf", "add_document"):
        if hasattr(retriever, method_name):
            fn = getattr(retriever, method_name)
            sig = inspect.signature(fn)
            if len(sig.parameters) >= 2:
                return fn(uploaded_file, course_tag)
            return fn(uploaded_file)

    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    new_chunks = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < 40:
            continue
        new_chunks.append(
            {
                "course": course_tag,
                "source": uploaded_file.name,
                "page": page_idx,
                "text": text,
                "uploaded": True,
            }
        )
    if new_chunks and hasattr(retriever, "chunks"):
        retriever.chunks.extend(new_chunks)
        if hasattr(retriever, "_rebuild_indices"):
            retriever._rebuild_indices()
    return len(new_chunks)


if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None
if "uploaded_files_list" not in st.session_state:
    st.session_state["uploaded_files_list"] = []

with st.sidebar:
    if LOGO_PATH:
        st.image(LOGO_PATH, width=115)

    lang_choice = st.segmented_control(
        "Language / 语言",
        options=["English", "中文"],
        default="English",
    )
    lang = "zh" if lang_choice == "中文" else "en"
    t = UI_TEXT[lang]

    st.markdown(f"### {t['controls']}")
    st.caption(t["controls_desc"])

    courses = get_available_courses(t["all_courses"])
    selected_course = st.selectbox(t["course_scope"], courses, index=0)
    course_filter = None if selected_course == t["all_courses"] else selected_course

    top_k = st.slider(t["top_k"], min_value=3, max_value=8, value=5)

    st.divider()
    st.markdown(f"### {t['upload_title']}")
    uploaded_pdf = st.file_uploader(t["upload_label"], type=["pdf"])
    custom_course = st.text_input(t["course_tag"], value=t["default_tag"])
    if st.button(t["index_btn"], type="primary", use_container_width=True, disabled=uploaded_pdf is None):
        with st.spinner(t["indexing"]):
            added = add_uploaded_pdf(uploaded_pdf, custom_course.strip() or t["default_tag"])
        st.session_state["uploaded_files_list"].append(uploaded_pdf.name)
        st.toast(f"{uploaded_pdf.name} (+{added} chunks)")
        st.rerun()

    if st.session_state["uploaded_files_list"]:
        st.caption(t["uploaded_caption"] + ", ".join(st.session_state["uploaded_files_list"]))

    st.divider()
    if st.button(t["clear_btn"], type="secondary", use_container_width=True):
        st.session_state["messages"] = []
        st.toast(t["cleared_toast"])
        st.rerun()


@st.dialog(t["dialog_title"])
def show_about_dialog():
    st.markdown(t["dialog_body"])
    st.info(t["dialog_tip"])


hero_logo = get_logo_html(width=82)
hero_title = "StarMate" if hero_logo else "⭐ StarMate"
st.markdown(
    f"""
    <div class="paper-hero">
        {hero_logo}
        <div>
            <div class="paper-eyebrow">{t['eyebrow']}</div>
            <h1>{hero_title}</h1>
            <p>{t['subtitle']}</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

ctrl_left, ctrl_right = st.columns([5, 1])
with ctrl_left:
    mode_label = st.segmented_control(
        "Retrieval Mode",
        options=["Hybrid (RRF)", "Vector Only", "BM25 Only"],
        default="Hybrid (RRF)",
        label_visibility="collapsed",
    )
with ctrl_right:
    if st.button(t["info_btn"], type="primary", use_container_width=True):
        show_about_dialog()

mode_map = {"Hybrid (RRF)": "hybrid", "Vector Only": "vector", "BM25 Only": "bm25"}
active_mode = mode_map.get(mode_label, "hybrid")

total_chunks = len(getattr(pipeline.retriever, "chunks", []))
total_courses = max(len(courses) - 1, 0)

st.markdown(
    f"""
    <div class="metric-grid">
        <div class="paper-metric">
            <div class="metric-label">{t['metric_chunks']}</div>
            <div class="metric-value">{total_chunks}</div>
        </div>
        <div class="paper-metric">
            <div class="metric-label">{t['metric_courses']}</div>
            <div class="metric-value">{total_courses}</div>
        </div>
        <div class="paper-metric">
            <div class="metric-label">{t['metric_mode']}</div>
            <div class="metric-value">{mode_label or "Hybrid (RRF)"}</div>
        </div>
        <div class="paper-metric">
            <div class="metric-label">{t['metric_guard']}</div>
            <div class="metric-value">{t['guard_val']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_sources(sources):
    if not sources:
        return
    with st.expander(t["sources_expander"].format(n=len(sources)), expanded=False):
        for idx, src in enumerate(sources, start=1):
            course = src.get("course", "Course")
            filename = src.get("source", "Unknown PDF")
            page = src.get("page", "?")
            snippet = src.get("text", "").strip()
            preview = snippet[:420] + ("..." if len(snippet) > 420 else "")
            st.markdown(
                f"<span class='course-badge'>{course}</span> **[{idx}] {filename} · Page {page}**",
                unsafe_allow_html=True,
            )
            st.caption(preview)
            if idx < len(sources):
                st.divider()


if not st.session_state["messages"]:
    st.markdown(t["starters_title"])
    c1, c2, c3 = st.columns(3)
    starters = [
        ("MIT 18.06 · Linear Algebra", "Explain the nullspace and column space of a matrix."),
        ("CS229 · Machine Learning", "How does gradient descent update parameters in linear regression?"),
        ("MIT 7.03 · Genetics", "Explain Mendelian inheritance and complementation tests."),
    ]
    for col, (title, q_text) in zip([c1, c2, c3], starters):
        with col:
            if st.button(f"{title}\n\n{q_text}", type="secondary", use_container_width=True):
                st.session_state["pending_prompt"] = q_text
                st.rerun()

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

user_input = st.chat_input(t["chat_placeholder"])
prompt = st.session_state.pop("pending_prompt", None) or user_input

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.time()
        with st.status(t["status_running"], expanded=True) as status:
            st.write(t["status_step1"].format(mode=mode_label or "Hybrid (RRF)"))
            st.write(t["status_step2"].format(course=selected_course, k=top_k))
            answer_text, sources = run_pipeline_answer(
                prompt, course_filter=course_filter, mode=active_mode, top_k=top_k, lang=lang
            )
            elapsed = time.time() - start_time
            st.write(t["status_step3"].format(n=len(sources)))
            status.update(label=t["status_done"].format(t=elapsed), state="complete", expanded=False)

        st.markdown(answer_text)
        render_sources(sources)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )