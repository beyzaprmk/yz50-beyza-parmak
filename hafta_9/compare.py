
from pathlib import Path
import importlib.util
import sys

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
WEEK8_DIR = PROJECT_ROOT / "hafta_8"

TOKENIZER_FILE = BASE_DIR / "tokenizer.py"

spec = importlib.util.spec_from_file_location(
    "week9_tokenizer",
    TOKENIZER_FILE,
)

if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load tokenizer from: {TOKENIZER_FILE}")

week9_tokenizer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = week9_tokenizer
spec.loader.exec_module(week9_tokenizer)

BPETokenizer = week9_tokenizer.BPETokenizer
Cl100kTokenizer = week9_tokenizer.Cl100kTokenizer


DATA_PATH = WEEK8_DIR / "data" / "kuyruksuz_tilki_clean.txt"

if not DATA_PATH.is_file():
    alternate = WEEK8_DIR / "data" / "kuyruksuz_tilki.txt"
    if alternate.is_file():
        DATA_PATH = alternate


VOCAB_SIZE = 280
TRAINING_SAMPLE_BYTES = 100_000


def train_own_bpe(text):
    sample_bytes = text.encode("utf-8")[:TRAINING_SAMPLE_BYTES]

    sample = sample_bytes.decode("utf-8", errors="ignore")

    tokenizer = BPETokenizer().train(
        sample,
        vocab_size=VOCAB_SIZE,
    )

    return tokenizer


def compare_text(label, text, own_tokenizer, cl100k):
    """Compare token counts and round-trip behavior."""
    own_ids = own_tokenizer.encode(text)
    cl100k_ids = cl100k.encode(text)

    byte_count = len(text.encode("utf-8"))
    own_count = len(own_ids)
    cl100k_count = len(cl100k_ids)

    print("Text:", text)
    print(f"UTF-8 bytes: {byte_count}")
    print(f"Own BPE token count: {own_count}")
    print(f"cl100k_base token count: {cl100k_count}")

    own_ratio = byte_count / own_count if own_count else 0
    cl100k_ratio = byte_count / cl100k_count if cl100k_count else 0

    print(f"Own BPE bytes/token: {own_ratio:.3f}")
    print(f"cl100k_base bytes/token: {cl100k_ratio:.3f}")

    print(
        "Own BPE round-trip:",
        own_tokenizer.decode(own_ids) == text,
    )
    print(
        "cl100k_base round-trip:",
        cl100k.decode(cl100k_ids) == text,
    )

    print("Own BPE IDs first 60:", own_ids[:60])
    print("cl100k_base IDs first 60:", cl100k_ids[:60])


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Turkish corpus not found: {DATA_PATH}\n"
            "Check the filename inside hafta_8/data/."
        )

    text = DATA_PATH.read_text(encoding="utf-8")

    own_tokenizer = train_own_bpe(text)

    cl100k = Cl100kTokenizer()

    print(f"Characters: {len(text):,}")
    print(f"UTF-8 bytes: {len(text.encode('utf-8')):,}")
    print(f"Own BPE vocab size: {own_tokenizer.vocab_size}")
    print(f"cl100k_base vocab size: {cl100k.vocab_size}")

    # Compare the complete corpus.
    own_stats = own_tokenizer.compression_stats(text)
    cl_stats = cl100k.compression_stats(text)

    print(f"Own BPE tokens: {own_stats['tokens']:,}")
    print(f"cl100k_base tokens: {cl_stats['tokens']:,}")
    print(
        "Own BPE bytes/token: "
        f"{own_stats['bytes_per_token']:.4f}"
    )
    print(
        "cl100k_base bytes/token: "
        f"{cl_stats['bytes_per_token']:.4f}"
    )


    for i, (pair, new_id) in enumerate(
        own_tokenizer.merges[:20],
        start=1,
    ):
        left = own_tokenizer.vocab[pair[0]]
        right = own_tokenizer.vocab[pair[1]]
        merged = own_tokenizer.vocab[new_id]

        print(
            f"{i:02d}. {left!r} + {right!r} "
            f"-> {merged!r}"
        )

    turkish = (
        "Ankara'dan İstanbul'a geldim. "
        "Türkçe öğreniyorum; çağrı, özgürlük ve "
        "düşünce önemlidir. 😊"
    )

    english = (
        "I came from Ankara to Istanbul. "
        "I am learning Turkish; freedom and "
        "thought are important. 😊"
    )

    compare_text(
        "Turkish sample",
        turkish,
        own_tokenizer,
        cl100k,
    )

    compare_text(
        "English counterpart",
        english,
        own_tokenizer,
        cl100k,
    )

    for label, sample in [
        ("Turkish", turkish),
        ("English", english),
    ]:
        word_count = len(sample.split())

        if word_count == 0:
            continue

        own_per_word = (
            len(own_tokenizer.encode(sample)) / word_count
        )
        cl100k_per_word = (
            len(cl100k.encode(sample)) / word_count
        )

        print(
            f"{label}: words={word_count}, "
            f"own BPE={own_per_word:.3f} tokens/word, "
            f"cl100k_base={cl100k_per_word:.3f} tokens/word"
        )


if __name__ == "__main__":
    main()
