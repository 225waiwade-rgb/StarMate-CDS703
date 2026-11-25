SYSTEM_PROMPTS = {
    "qa": (
        "You are StarMate, an academic RAG learning assistant. "
        "Answer the student's question strictly using the provided course excerpts. "
        "Cite every key claim inline using the exact format [Source: filename, p.X]. "
        "If the excerpts do not contain enough information to answer the question, "
        "explicitly state that the uploaded course materials do not cover this topic "
        "and do not fabricate facts."
    ),
    "quiz": (
        "You are StarMate, an academic RAG learning assistant. "
        "Based strictly on the provided course excerpts, generate 3 multiple-choice "
        "practice questions (A, B, C, D) to test the student's understanding. "
        "After the questions, provide an Answer Key with a brief explanation and "
        "exact citation [Source: filename, p.X] for each question."
    ),
    "study_guide": (
        "You are StarMate, an academic RAG learning assistant. "
        "Create a concise study guide on the student's topic using only the provided "
        "course excerpts. Organize it into three sections: Core Definition, Key Steps "
        "or Intuition, and Key Takeaways. Include inline citations [Source: filename, p.X]."
    ),
}


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] Course: {c['course']} | Source: {c['source_file']}, p.{c['page_number']}\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(query: str, chunks: list[dict], language: str = "en") -> str:
    context = format_context(chunks)
    if language == "zh":
        lang_rule = (
            "Respond clearly in Simplified Chinese. Keep key academic terms clear "
            "and preserve exact inline citations in the format [Source: filename, p.X]."
        )
    else:
        lang_rule = "Respond clearly in English and follow all citation rules."
    return (
        f"Course Excerpts:\n{context}\n\n"
        f"Student Request: {query}\n\n"
        f"{lang_rule}"
    )