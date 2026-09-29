from layers import (
    Embedding,
    Flatten,
    FlattenConsecutive,
    Linear,
    BatchNorm1d,
    BuggyBatchNorm1d,
    Tanh,
    Sequential
)

class MLP:

    def __init__(
        self,
        vocab_size,
        block_size=3,
        embedding_dim=2,
        hidden_dim=100
    ):

        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.model = Sequential(

            Embedding(
                vocab_size,
                embedding_dim
            ),

            Flatten(),

            Linear(
                block_size * embedding_dim,
                hidden_dim
            ),

            BatchNorm1d(
                hidden_dim
            ),

            Tanh(),

            Linear(
                hidden_dim,
                vocab_size
            )
        )

    def __call__(self, x):

        return self.model(x)

    def parameters(self):

        return self.model.parameters()

    def train(self):

        self.model.train()

    def eval(self):

        self.model.eval()

class WaveNet:

    def __init__(
        self,
        vocab_size,
        block_size=8,
        embedding_dim=2,
        hidden_dim=100
    ):

        if block_size != 8:

            raise ValueError(
                "WaveNet requires block_size=8."
            )

        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.model = Sequential(

            # [B, 8, E]
            Embedding(
                vocab_size,
                embedding_dim
            ),

            # [B, 4, 2E]
            FlattenConsecutive(2),

            Linear(
                2 * embedding_dim,
                hidden_dim
            ),

            BatchNorm1d(
                hidden_dim
            ),

            Tanh(),

            # [B, 2, 2H]
            FlattenConsecutive(2),

            Linear(
                2 * hidden_dim,
                hidden_dim
            ),

            BatchNorm1d(
                hidden_dim
            ),

            Tanh(),

            # [B, 1, 2H]
            FlattenConsecutive(2),

            Linear(
                2 * hidden_dim,
                hidden_dim
            ),

            BatchNorm1d(
                hidden_dim
            ),

            Tanh(),

            # [B, 1, vocab]
            Linear(
                hidden_dim,
                vocab_size
            )
        )

    def __call__(self, x):

        return self.model(x)

    def parameters(self):

        return self.model.parameters()

    def train(self):

        self.model.train()

    def eval(self):

        self.model.eval()


class WaveNetBuggyBN:

    def __init__(
        self,
        vocab_size,
        block_size=8,
        embedding_dim=2,
        hidden_dim=100
    ):

        if block_size != 8:

            raise ValueError(
                "WaveNet requires block_size=8."
            )

        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.model = Sequential(

            Embedding(
                vocab_size,
                embedding_dim
            ),

            FlattenConsecutive(2),

            Linear(
                2 * embedding_dim,
                hidden_dim
            ),

            BuggyBatchNorm1d(
                hidden_dim
            ),

            Tanh(),

            FlattenConsecutive(2),

            Linear(
                2 * hidden_dim,
                hidden_dim
            ),

            BuggyBatchNorm1d(
                hidden_dim
            ),

            Tanh(),

            FlattenConsecutive(2),

            Linear(
                2 * hidden_dim,
                hidden_dim
            ),

            BuggyBatchNorm1d(
                hidden_dim
            ),

            Tanh(),

            Linear(
                hidden_dim,
                vocab_size
            )
        )

    def __call__(self, x):

        return self.model(x)

    def parameters(self):

        return self.model.parameters()

    def train(self):

        self.model.train()

    def eval(self):

        self.model.eval()

def count_parameters(model):

    return sum(
        p.nelement()
        for p in model.parameters()
    )