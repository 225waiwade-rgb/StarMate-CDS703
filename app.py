import io
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from ingest import recursive_split
from pipeline import StarMatePipeline


st.set_page_config(
    page_title="StarMate | Course RAG Assistant",
    page_icon="⭐",
    layout="wide",
)


def check_access_password() -> None:
    expected = os.getenv("APP_PASSWORD", "").strip()
    if not expected or st.session_state.get("authenticated"):
        return

    st.title("⭐ StarMate Access / 访问验证")
    st.caption("Please enter the course access code to continue. / 请输入课程访问口令。")
    pwd = st.text_input("Access Code / 访问口令", type="password")
    if st.button("Enter / 进入系统"):
        if pwd == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect access code. / 访问口令错误，请重试。")
    st.stop()


check_access_password()

UI_TEXT = {
    "en": {
        "sidebar_title": "⭐ StarMate Settings",
        "sidebar_caption": "CDS703 Course-Aware RAG Learning Assistant",
        "upload_header": "Upload Your Own PDFs",
        "upload_course_label": "Course Tag for Uploaded Files",
        "upload_course_default": "My Uploaded Notes",
        "upload_label": "Choose PDF lecture notes",
        "upload_btn": "Process & Add to Knowledge Base",
        "upload_reset_btn": "Remove Uploaded Files",
        "upload_success": "Added {n} chunks from {files} file(s) under '{course}'.",
        "upload_empty": "No extractable text found (the PDF may be a scanned image).",
        "upload_dup": "Selected file(s) are already indexed.",
        "course_label": "Course Filter",
        "all_courses": "All Courses",
        "mode_label": "Learning Mode",
        "modes": {
            "Q&A (Cited)": "qa",
            "Practice Quiz (3 MCQs)": "quiz",
            "Study Guide": "study_guide",
        },
        "ret_label": "Retrieval Method",
        "ret_modes": {
            "Hybrid (BM25 + Vector)": "hybrid",
            "BM25 Only (Keyword)": "bm25",
            "Vector Only (Semantic)": "vector",
        },
        "topk_label": "Top-K Excerpts",
        "chunks_stat": "Indexed Chunks",
        "courses_stat": "Courses",
        "clear_btn": "Clear Conversation",
        "main_title": "⭐ StarMate: Academic RAG Assistant",
        "main_caption": (
            "Ask course questions, generate practice quizzes, or build study guides "
            "grounded strictly in built-in or uploaded lecture PDFs."
        ),
        "samples": [
            "How does Gaussian elimination work?",
            "Explain Mendel's law of segregation.",
            "What is gradient clipping and why is it used?",
        ],
        "input_placeholder": "Ask a question about your course materials...",
        "spinner": "Searching lecture notes and generating answer...",
        "sources_title": "View Retrieved Sources",
        "translated_note": "English retrieval query used",
    },
    "zh": {
        "sidebar_title": "⭐ StarMate 设置",
        "sidebar_caption": "CDS703 课程资料 RAG 学习助手",
        "upload_header": "上传个人 PDF 课件",
        "upload_course_label": "为上传课件设置课程分类名",
        "upload_course_default": "My Uploaded Notes",
        "upload_label": "选择本地 PDF 讲义或笔记（可多选）",
        "upload_btn": "解析并加入知识库",
        "upload_reset_btn": "清除已上传课件",
        "upload_success": "已成功从 {files} 份文件解析 {n} 个文本块，归入「{course}」。",
        "upload_empty": "未能从该 PDF 提取文字（可能是纯图片扫描件）。",
        "upload_dup": "所选文件已在知识库中，无需重复解析。",
        "course_label": "课程筛选",
        "all_courses": "全部课程",
        "mode_label": "学习模式",
        "modes": {
            "课程问答（带引用）": "qa",
            "练习测验（3道选择题）": "quiz",
            "知识点学习指南": "study_guide",
        },
        "ret_label": "检索方式",
        "ret_modes": {
            "混合检索（BM25 + 向量）": "hybrid",
            "仅关键词检索（BM25）": "bm25",
            "仅语义向量检索（Vector）": "vector",
        },
        "topk_label": "召回片段数（Top-K）",
        "chunks_stat": "已索引文本块",
        "courses_stat": "课程数量",
        "clear_btn": "清空对话记录",
        "main_title": "⭐ StarMate：课程知识库 RAG 学习助手",
        "main_caption": "严格基于内置或个人上传的 PDF 讲义，提供带页码溯源的课程问答、自动出题与复习指南。",
        "samples": [
            "高斯消元法是如何工作的？",
            "请解释孟德尔的分离定律。",
            "什么是梯度裁剪，为什么要使用它？",
        ],
        "input_placeholder": "请输入关于课程讲义的问题...",
        "spinner": "正在检索讲义并生成回答...",
        "sources_title": "查看检索来源",
        "translated_note": "实际用于检索英文讲义的查询词",
    },
}


def format_latex(text: str) -> str:
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return text


def parse_uploaded_pdf(uploaded_file, course_name: str) -> list[dict]:
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    stem = Path(uploaded_file.name).stem
    chunks = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        pieces = recursive_split(text, chunk_size=800, chunk_overlap=80)
        for chunk_idx, piece in enumerate(pieces, start=1):
            safe_id = f"{course_name}_{stem}_p{page_idx}_c{chunk_idx}".replace(" ", "_")
            chunks.append(
                {
                    "chunk_id": safe_id,
                    "course": course_name,
                    "source_file": uploaded_file.name,
                    "page_number": page_idx,
                    "text": piece,
                }
            )
    return chunks


@st.cache_resource
def load_pipeline() -> StarMatePipeline:
    return StarMatePipeline()


pipeline = load_pipeline()

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

with st.sidebar:
    lang_choice = st.radio(
        "Language / 语言",
        ["English", "中文"],
        index=0,
        horizontal=True,
    )
    lang = "en" if lang_choice == "English" else "zh"
    t = UI_TEXT[lang]

    st.title(t["sidebar_title"])
    st.caption(t["sidebar_caption"])

    with st.expander(f"📄 {t['upload_header']}", expanded=False):
        custom_course = st.text_input(
            t["upload_course_label"], value=t["upload_course_default"]
        ).strip() or t["upload_course_default"]
        uploaded = st.file_uploader(
            t["upload_label"], type=["pdf"], accept_multiple_files=True
        )
        if uploaded and st.button(t["upload_btn"], use_container_width=True):
            new_chunks = []
            added_files = 0
            for uf in uploaded:
                file_key = f"{custom_course}::{uf.name}"
                if file_key in st.session_state.uploaded_files:
                    continue
                parsed = parse_uploaded_pdf(uf, custom_course)
                if parsed:
                    new_chunks.extend(parsed)
                    st.session_state.uploaded_files.append(file_key)
                    added_files += 1
            if new_chunks:
                n_added = pipeline.retriever.add_chunks(new_chunks)
                st.success(
                    t["upload_success"].format(
                        n=n_added, files=added_files, course=custom_course
                    )
                )
            elif added_files == 0 and all(
                f"{custom_course}::{uf.name}" in st.session_state.uploaded_files
                for uf in uploaded
            ):
                st.info(t["upload_dup"])
            else:
                st.warning(t["upload_empty"])

        if st.session_state.uploaded_files:
            if st.button(t["upload_reset_btn"], use_container_width=True):
                pipeline.retriever.reset_to_base()
                st.session_state.uploaded_files = []
                st.rerun()

    available_courses = sorted({c["course"] for c in pipeline.retriever.chunks})
    course_options = [t["all_courses"]] + available_courses
    selected_course = st.selectbox(t["course_label"], course_options)
    course = "All" if selected_course == t["all_courses"] else selected_course

    selected_mode_label = st.selectbox(t["mode_label"], list(t["modes"].keys()))
    task_mode = t["modes"][selected_mode_label]

    selected_ret_label = st.selectbox(t["ret_label"], list(t["ret_modes"].keys()))
    retrieval_mode = t["ret_modes"][selected_ret_label]

    top_k = st.slider(t["topk_label"], min_value=2, max_value=8, value=5)

    st.divider()
    st.markdown(f"**{t['chunks_stat']}:** {len(pipeline.retriever.chunks)}")
    st.markdown(f"**{t['courses_stat']}:** {len(available_courses)}")

    if st.button(t["clear_btn"], use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title(t["main_title"])
st.caption(t["main_caption"])

if "messages" not in st.session_state:
    st.session_state.messages = []

cols = st.columns(len(t["samples"]))
clicked_query = None
for col, sq in zip(cols, t["samples"]):
    if col.button(sq, use_container_width=True):
        clicked_query = sq

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(format_latex(msg["content"]))
        if msg.get("sources"):
            with st.expander(f"{t['sources_title']} ({len(msg['sources'])})"):
                if msg.get("search_query") and msg["search_query"] != msg.get("original_query"):
                    st.info(f"{t['translated_note']}: `{msg['search_query']}`")
                for i, s in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f"**{i}. {s['course']}** — `{s['source_file']}` (p.{s['page_number']}, score={s['score']})"
                    )
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

user_input = st.chat_input(t["input_placeholder"])
prompt = clicked_query or user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(t["spinner"]):
            result = pipeline.answer(
                query=prompt,
                course=course,
                task_mode=task_mode,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                language=lang,
            )
        answer_text = result["answer"]
        sources = result["sources"]
        search_query = result.get("search_query", prompt)
        st.markdown(format_latex(answer_text))

        if sources:
            with st.expander(f"{t['sources_title']} ({len(sources)})"):
                if search_query != prompt:
                    st.info(f"{t['translated_note']}: `{search_query}`")
                for i, s in enumerate(sources, start=1):
                    st.markdown(
                        f"**{i}. {s['course']}** — `{s['source_file']}` (p.{s['page_number']}, score={s['score']})"
                    )
                    st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer_text,
            "sources": sources,
            "search_query": search_query,
            "original_query": prompt,
        }
    )