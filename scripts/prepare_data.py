
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXTRACT_DIR = DATA_DIR / "asia_documents"
ZIP_PATH = DATA_DIR / "asia_documents.zip"

DATASET_URL = (
    # "https://github.com/gakudo-ai/open-datasets/"
    # "blob/main/asia_documents.zip?raw=true"
    "https://github.com/gakudo-ai/open-datasets/blob/main/asia_documents.zip?raw=true"
)


def prepare_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    if not EXTRACT_DIR.exists():
        EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(EXTRACT_DIR)

    documents = []

    for filepath in EXTRACT_DIR.rglob("*.txt"):
        text = filepath.read_text(encoding="utf-8").strip()

        if text:
            documents.append(text)

    if not documents:
        raise RuntimeError(
            f"No text documents found in {EXTRACT_DIR}"
        )

    print(f"Loaded {len(documents)} documents.")
    return documents


if __name__ == "__main__":
    prepare_data()
