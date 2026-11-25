import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parent))
from retriever import HybridRetriever
from prompts import SYSTEM_PROMPTS, build_user_prompt

load_dotenv()


class StarMatePipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        self.model = os.getenv("LLM_MODEL", "deepseek-flash")

        if not api_key:
            raise ValueError("Missing LLM_API_KEY in .env file.")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def _prepare_search_query(self, query: str) -> str:
        if not re.search(r"[\u4e00-\u9fff]", query):
            return query
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Translate the following academic question into concise English "
                            "for searching English university lecture notes. Output only the "
                            "English translation."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
            )
            translated = (resp.choices[0].message.content or "").strip()
            return translated or query
        except Exception:
            return query

    def answer(
        self,
        query: str,
        course: str = "All",
        task_mode: str = "qa",
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        language: str = "en",
    ) -> dict:
        search_query = self._prepare_search_query(query)
        chunks = self.retriever.search(
            query=search_query,
            course=course,
            top_k=top_k,
            mode=retrieval_mode,
        )
        if not chunks:
            empty_msg = (
                "未在已上传的课程讲义中找到相关内容。"
                if language == "zh"
                else "No relevant course excerpts were found for your query."
            )
            return {"answer": empty_msg, "sources": [], "search_query": search_query}

        system_prompt = SYSTEM_PROMPTS.get(task_mode, SYSTEM_PROMPTS["qa"])
        user_prompt = build_user_prompt(query, chunks, language=language)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": chunks,
            "search_query": search_query,
        }