from __future__ import annotations

import pickle
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()


def create_file_table_df() -> pd.DataFrame:
    columns = ["file_id", "file_name", "file_path"]
    return pd.DataFrame(columns=columns)


def create_page_table_df() -> pd.DataFrame:
    columns = ["page_id", "file_id", "page_number", "text"]
    return pd.DataFrame(columns=columns)


def create_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    file_cache_path = cache_dir / "file_table.pkl"
    page_cache_path = cache_dir / "page_table.pkl"

    if file_cache_path.exists() and page_cache_path.exists():
        with open(file_cache_path, "rb") as f:
            cached_file_df: pd.DataFrame = pickle.load(f)
        with open(page_cache_path, "rb") as f:
            cached_page_df: pd.DataFrame = pickle.load(f)
        print("✅ キャッシュから file_table_df と page_table_df を読み込みました。")
        return cached_file_df, cached_page_df
    else:
        new_file_table_df = create_file_table_df()
        new_page_table_df = create_page_table_df()

        pdf_files = list(input_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            file_id = str(uuid.uuid4())
            new_file_table_df = append_to_file_table(
                new_file_table_df, file_id, pdf_file
            )
            texts = extract_text_from_pdf(pdf_file)
            for i, text in enumerate(texts):
                new_page_table_df = append_to_page_table(
                    new_page_table_df, i, file_id, text
                )

        with open(file_cache_path, "wb") as f:
            pickle.dump(new_file_table_df, f)
            print(f"✅ データをキャッシュに保存しました: {file_cache_path.name}")
        with open(page_cache_path, "wb") as f:
            pickle.dump(new_page_table_df, f)
            print(f"✅ データをキャッシュに保存しました: {page_cache_path.name}")

        return new_file_table_df, new_page_table_df


def extract_text_from_pdf(_file_path: Path) -> list[str]:
    reader = PdfReader(str(_file_path))
    return [page.extract_text() or "" for page in reader.pages]


def append_to_file_table(
    df: pd.DataFrame, _file_id: str, _file_path: Path
) -> pd.DataFrame:
    new_data = {
        "file_id": [_file_id],
        "file_name": [_file_path.name],
        "file_path": [str(_file_path)],
    }
    return pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)


def append_to_page_table(
    df: pd.DataFrame, _index: int, _file_id: str, _text: str
) -> pd.DataFrame:
    new_data = {
        "page_id": [_file_id + "_" + str(_index)],
        "file_id": [_file_id],
        "page_number": [_index + 1],
        "text": [_text],
    }
    return pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)


if __name__ == "__main__":
    input_dir = Path("./input")
    output_dir = Path("./output")
    cache_dir = Path("./cache")
    now = datetime.now()

    assert input_dir.exists(), f"{input_dir} does not exist"
    output_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    file_table_df, page_table_df = create_dataframes()
    file_table_df.to_csv(output_dir / "file_table.csv", index=False)
    page_table_df.to_csv(output_dir / "page_table.csv", index=False)
