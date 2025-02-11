from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    GPTVectorStoreIndex,
    StorageContext,
)
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    file_table = pd.read_csv("output/file_table.csv")
    page_table = pd.read_csv("output/page_table.csv")

    documents = [
        Document(
            text=row["text"],
            metadata={
                "file_id": row["file_id"],
                "page_number": row["page_number"],
                "summary": row.get("summary", ""),
                "tags": (
                    row.get("tags", "").split(",") if pd.notna(row.get("tags")) else []
                ),
            },
        )
        for _, row in page_table.iterrows()
    ]

    embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")
    llm_model = OpenAI(model="gpt-4o-mini")
    prompt_helper = PromptHelper(
        context_window=4096,  # LLMが一度に処理できる最大トークン数（約4000単語程度）
        num_output=256,  # 生成される回答の最大トークン数（長文回答の制御）
        chunk_overlap_ratio=0.05,  # テキストを分割する際の「重なり率」（重要な文脈が失われないようにするため）
    )
    storage_context = StorageContext.from_defaults()  # ローカルに保存
    index = GPTVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    index.storage_context.persist("output/index")
