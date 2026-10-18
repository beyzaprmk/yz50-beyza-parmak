
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.token_to_id = {bytes([i]): i for i in range(256)}
        self.vocab_size = 256
        self.trained = False

    @staticmethod
    def _count_pairs(ids):
        counts = Counter()
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
        return counts

    @staticmethod
    def _merge_pair(ids, pair, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text, vocab_size=512, verbose=False):
        if not isinstance(text, str) or not text:
            raise ValueError("Training text must be a non-empty string.")
        if vocab_size < 256:
            raise ValueError("Byte-level BPE vocab_size must be at least 256.")

        ids = list(text.encode("utf-8"))
        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.token_to_id = {bytes([i]): i for i in range(256)}

        target_size = min(vocab_size, 256 + max(0, len(ids) - 1))
        while len(self.vocab) < target_size:
            counts = self._count_pairs(ids)
            if not counts:
                break
            pair, count = counts.most_common(1)[0]
            if count < 2:
                break
            new_id = len(self.vocab)
            merged_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges.append((pair, new_id))
            self.vocab[new_id] = merged_bytes
            self.token_to_id[merged_bytes] = new_id
            ids = self._merge_pair(ids, pair, new_id)
            if verbose and len(self.merges) <= 20:
                print(
                    f"merge {len(self.merges):02d}: "
                    f"{self.vocab[pair[0]]!r} + {self.vocab[pair[1]]!r} "
                    f"-> {merged_bytes!r} (count={count})"
                )

        self.vocab_size = len(self.vocab)
        self.trained = True
        return self

    def encode(self, text):
        if not self.trained:
            raise RuntimeError("Call train() or load() before encode().")
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges:
            ids = self._merge_pair(ids, pair, new_id)
        return ids

    def decode(self, ids):
        raw = b"".join(self.vocab[int(token_id)] for token_id in ids)
        return raw.decode("utf-8", errors="replace")

    def compression_stats(self, text):
        raw_bytes = len(text.encode("utf-8"))
        token_count = len(self.encode(text))
        return {
            "bytes": raw_bytes,
            "tokens": token_count,
            "bytes_per_token": raw_bytes / token_count if token_count else 0.0,
        }

    def save(self, path):
        import json
        payload = {
            "vocab_size": self.vocab_size,
            "merges": [[list(pair), new_id] for pair, new_id in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tokenizer = cls()
        tokenizer.merges = [
            ((int(pair[0]), int(pair[1])), int(new_id))
            for pair, new_id in payload["merges"]
        ]
        tokenizer.vocab = {i: bytes([i]) for i in range(256)}
        for pair, new_id in tokenizer.merges:
            tokenizer.vocab[new_id] = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
        tokenizer.token_to_id = {value: key for key, value in tokenizer.vocab.items()}
        tokenizer.vocab_size = len(tokenizer.vocab)
        tokenizer.trained = True
        return tokenizer


class Cl100kTokenizer:
    
    def __init__(self):
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "Cl100kTokenizer requires tiktoken. Install it with "
                "'python -m pip install tiktoken'."
            ) from exc
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.name = "cl100k_base"

    @property
    def vocab_size(self):
        return self.encoding.n_vocab

    def encode(self, text):
        return self.encoding.encode(text, disallowed_special=())


    def decode(self, ids):
        return self.encoding.decode(list(ids))

    def compression_stats(self, text):
        raw_bytes = len(text.encode("utf-8"))
        token_count = len(self.encode(text))
        return {
            "bytes": raw_bytes,
            "tokens": token_count,
            "bytes_per_token": raw_bytes / token_count if token_count else 0.0,
        }
