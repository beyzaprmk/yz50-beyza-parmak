
class CharacterTokenizer:
    def __init__(self, text):
        if not text:
            raise ValueError("Tokenizer cannot be built from empty text.")

        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            preview = "".join(unknown[:10])
            raise ValueError(
                f" characters not present in tokenizer vocabulary: {preview!r}"
            )
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(index)] for index in ids)
