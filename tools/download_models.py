"""Fetch the embedding and reranker weights this project runs against.

Every model is pinned to a commit hash rather than a tag or branch. Hub
revisions move, and an encoder whose weights changed silently invalidates every
vector already written to the stores -- old and new vectors would occupy
different spaces while still comparing without error. Pinning makes the
embedding space part of the reproducible configuration.

Run once after cloning; `grace_mem/adapters/embedding/embeddings.py` falls back to downloading
from the Hub if these are absent, but then the revision is whatever is current.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

# Models live at the repository root, even though this helper is under tools/.
REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = REPO_ROOT / "models"

MODELS = {
    "Qwen/Qwen3-Embedding-0.6B": (
        TARGET_DIR / "embedding_models" / "qwen3-0.6b",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    ),
    "Qwen/Qwen3-Reranker-0.6B": (
        TARGET_DIR / "reranker" / "qwen3-reranker-0.6b",
        "e61197ed45024b0ed8a2d74b80b4d909f1255473",
    ),
}

def main():
    """Download every pinned model into models/, skipping what is already there.

    `snapshot_download` is incremental, so re-running costs a metadata check
    rather than a re-download.
    """
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id, (local_dir, revision) in MODELS.items():
        print(f"Downloading {repo_id}@{revision} -> {local_dir}")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            # Copy real files instead of symlinking into the HF cache, so the
            # models/ tree can be archived or moved to an offline machine on
            # its own.
            local_dir_use_symlinks=False,
            revision=revision,
            # Skip the .pt duplicates -- safetensors covers what is loaded, and
            # the .pt copies roughly double the download for nothing.
            ignore_patterns=["*.pt", "*.bin.tmp"],
        )
    print("All done.")

if __name__ == "__main__":
    main()
