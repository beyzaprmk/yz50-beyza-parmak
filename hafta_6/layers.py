import torch


class Linear:

    def __init__(self, fan_in, fan_out):

        self.weight = (
            torch.randn(fan_in, fan_out) * 0.1
        )

        self.bias = (
            torch.randn(fan_out) * 0.1
        )

        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def __call__(self, x):

        return x @ self.weight + self.bias

    def parameters(self):

        return [
            self.weight,
            self.bias
        ]

class BatchNorm1d:

    def __init__(
        self,
        dim,
        momentum=0.1,
        eps=1e-5
    ):

        self.gamma = torch.ones(dim, requires_grad=True)

        self.beta = torch.zeros( dim,  requires_grad=True )

        self.running_mean = torch.zeros(  1,  dim )

        self.running_var = torch.ones(
            1,
            dim
        )

        self.momentum = momentum
        self.eps = eps
        self.training = True

    def __call__(self, x):

        if x.ndim == 2:

            x_2d = x

            original_shape = None

        elif x.ndim == 3:

            B, T, C = x.shape

            x_2d = x.reshape(   B * T, C)

            original_shape = (  B,T, C)

        else:

            raise ValueError(
                "BatchNorm1d expects "
                "[B, C] or [B, T, C]"
            )

        if self.training:

            batch_mean = x_2d.mean(
                dim=0,
                keepdim=True
            )

            batch_var = x_2d.var(dim=0, keepdim=True,unbiased=False)

            x_centered = ( x_2d - batch_mean)

            inv_std = 1.0 / torch.sqrt( batch_var + self.eps)

            x_norm = (x_centered * inv_std)

            # Running statistics
            with torch.no_grad():

                self.running_mean.mul_(    1 - self.momentum ).add_(  self.momentum * batch_mean)

                self.running_var.mul_( 1 - self.momentum).add_(  self.momentum * batch_var)

        
        else:

            x_norm = (  x_2d - self.running_mean ) / torch.sqrt( self.running_var + self.eps )

        
        out = (
            self.gamma * x_norm
            + self.beta
        )

        if original_shape is not None:

            out = out.reshape(
                original_shape
            )

        return out

    def parameters(self):

        return [
            self.gamma,
            self.beta
        ]

class BuggyBatchNorm1d:

    def __init__(
        self,
        dim,
        momentum=0.1,
        eps=1e-5
    ):

        self.gamma = torch.ones(dim, requires_grad=True)

        self.beta = torch.zeros(dim,requires_grad=True )

        self.running_mean = torch.zeros(1, 1,dim)

        self.running_var = torch.ones( 1,1,dim)

        self.momentum = momentum
        self.eps = eps
        self.training = True

    def __call__(self, x):

        if x.ndim != 3:

            raise ValueError(
                "BuggyBatchNorm1d expects "
                "[B, T, C]"
            )
        if self.training:

            batch_mean = x.mean( dim=0, keepdim=True)

            batch_var = x.var(dim=0, keepdim=True,unbiased=False
            )

            x_norm = (
                x - batch_mean
            ) / torch.sqrt(
                batch_var + self.eps
            )

            with torch.no_grad():

                mean_for_running = (
                    batch_mean.mean(
                        dim=1,
                        keepdim=True
                    ) )

                var_for_running = (batch_var.mean(dim=1,keepdim=True)
                )

                self.running_mean.mul_(
                    1 - self.momentum
                ).add_(
                    self.momentum * mean_for_running
                )

                self.running_var.mul_(
                    1 - self.momentum
                ).add_(
                    self.momentum * var_for_running
                )

        else:
            x_norm = (
                x - self.running_mean
            ) / torch.sqrt(
                self.running_var + self.eps
            )

        return (self.gamma * x_norm+ self.beta)

    def parameters(self):

        return [self.gamma,self.beta]


class Tanh:

    def __call__(self, x):

        return torch.tanh(x)

    def parameters(self):

        return []


class Embedding:

    def __init__(
        self,
        num_embeddings,
        embedding_dim
    ):

        self.weight = torch.randn(
            num_embeddings,
            embedding_dim
        )

        self.weight.requires_grad = True

    def __call__(self, x):

        return self.weight[x]

    def parameters(self):

        return [
            self.weight
        ]



# [B, T, C] -> [B, T*C]

class Flatten:

    def __call__(self, x):

        return x.reshape(
            x.shape[0],
            -1
        )

    def parameters(self):

        return []


class FlattenConsecutive:

    def __init__(self, n):

        self.n = n

    def __call__(self, x):

        B, T, C = x.shape

        if T % self.n != 0:

            raise ValueError(
                f"Sequence length {T} "
                f"is not divisible by {self.n}"
            )

        return x.reshape(
            B,
            T // self.n,
            C * self.n
        )

    def parameters(self):

        return []

class Sequential:

    def __init__(self, *layers):

        self.layers = list(layers)

    def __call__(self, x):

        for layer in self.layers:

            x = layer(x)

        return x

    def parameters(self):

        parameters = []

        for layer in self.layers:

            parameters.extend(
                layer.parameters()
            )

        return parameters

    def train(self):

        for layer in self.layers:

            if hasattr( layer, "training"):

                layer.training = True

    def eval(self):

        for layer in self.layers:

            if hasattr(  layer,  "training" ):

                layer.training = False