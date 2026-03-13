from __future__ import annotations

from typing import Any

try:
    from langchain.chains import RetrievalQA  # type: ignore[import-not-found]
    from langchain.prompts import PromptTemplate  # type: ignore[import-not-found]
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    RetrievalQA = None  # type: ignore[assignment]
    PromptTemplate = None  # type: ignore[assignment]
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]

from .config import RAGSettings


class RAGQASystem:
    def __init__(self, settings: RAGSettings, logger):
        if RetrievalQA is None or PromptTemplate is None or ChatGoogleGenerativeAI is None:
            raise ImportError(
                "QA dependencies are missing. Install requirements.txt before running queries."
            )
        self.settings = settings
        self.logger = logger
        self.prompt = self._build_prompt()
        self._llm: Any | None = None

    def _build_prompt(self) -> Any:
        assert PromptTemplate is not None
        template = """You are an academic assistant for university students.
Use only the provided context to answer the question.

Context:
{context}

Question:
{question}

Rules:
1. If the question is in Vietnamese, answer in Vietnamese.
2. Do not invent facts that are not present in the context.
3. If evidence is missing, say: 'Khong tim thay thong tin nay trong tai lieu.'
4. Cite concise source clues from context when possible.
"""
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def _create_llm(self) -> Any:
        if self._llm is not None:
            return self._llm

        if not self.settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing. Add it to .env before running query."
            )

        assert ChatGoogleGenerativeAI is not None
        self._llm = ChatGoogleGenerativeAI(
            model=self.settings.llm_model,
            google_api_key=self.settings.google_api_key,
            temperature=self.settings.llm_temperature,
            max_output_tokens=self.settings.llm_max_tokens,
            convert_system_message_to_human=True,
        )
        return self._llm

    def query(self, question: str, vector_store, top_k: int | None = None) -> dict[str, Any]:
        llm = self._create_llm()
        effective_k = top_k or self.settings.retrieval_k
        assert RetrievalQA is not None

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": effective_k}),
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True,
        )

        self.logger.info("Running QA query | top_k=%s | question=%s", effective_k, question)
        result = chain.invoke({"query": question})

        sources = []
        for item in result.get("source_documents", []):
            metadata = item.metadata or {}
            sources.append(
                {
                    "source": metadata.get("source", "unknown"),
                    "chunk_id": metadata.get("chunk_id", -1),
                    "preview": item.page_content[:200],
                }
            )

        return {
            "question": question,
            "answer": result.get("result", ""),
            "sources": sources,
        }
