from pathlib import Path

from huggingface_hub import snapshot_download

# Models live at the repository root, even though this helper is under tools/.
REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = REPO_ROOT / "models"

MODELS = {
    # # Chat 模型（需有存取權）
    # "meta-llama/Llama-3.2-3B-Instruct": TARGET_DIR / "Llama-3.2-3B-Instruct",
    # # 英文通用 embedding 模型
    # "sentence-transformers/all-MiniLM-L6-v2": TARGET_DIR / "embedding_models" / "all-MiniLM-L6-v2",
    # # 中文小型 embedding（sentence-transformers 版）
    # "BAAI/bge-small-zh-v1.5": TARGET_DIR / "embedding_models" / "bge-small-zh-v1.5",
    # 多語單一模型（開源，本地化方便）
    # "BAAI/bge-m3": TARGET_DIR / "embedding_models" / "bge-m3",
    
    ## 新embedding model
    "Qwen/Qwen3-Embedding-0.6B": TARGET_DIR / "embedding_models" / "qwen3-0.6b",
    "Qwen/Qwen3-Reranker-0.6B": TARGET_DIR / "reranker" / "qwen3-reranker-0.6b"
    # "tomaarsen/Qwen3-Reranker-0.6B-seq-cls": TARGET_DIR / "reranker" / "qwen3-reranker-0.6b-seq-cls",
}

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id, local_dir in MODELS.items():
        print(f"Downloading {repo_id} -> {local_dir}")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # 複製實體檔，方便封裝/搬移
            revision=None,                 # 需要固定版本可填 commit hash/tag
            ignore_patterns=["*.pt", "*.bin.tmp"],  # 可選擇性忽略暫存
        )
    print("All done.")

if __name__ == "__main__":
    main()
