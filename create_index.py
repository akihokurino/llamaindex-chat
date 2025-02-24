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

    documents = [Document(text=row["text"]) for _, row in page_table.iterrows()]

    embedding_model = OpenAIEmbedding(model="text-embedding-ada-002")
    llm_model = OpenAI(model="gpt-4o-mini")
    prompt_helper = PromptHelper(
        context_window=8192,  # LLM（大規模言語モデル）が一度に処理できる 最大トークン数（入力＋出力の合計）
        num_output=512,  # LLMが生成する 最大出力トークン数
        chunk_overlap_ratio=0.5,  # テキストを分割する際の「重なり率」（重要な文脈が失われないようにするため）
    )
    storage_context = StorageContext.from_defaults()  # ローカルに保存
    index = GPTVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    index.storage_context.persist("output/index")
