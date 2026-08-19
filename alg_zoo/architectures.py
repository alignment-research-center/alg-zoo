import torch as th


class OneLayerRNN(th.nn.Module):
    def __init__(self, hidden_size, output_size, bias=False):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bias = bias
        super().__init__()
        self.rnn = th.nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            nonlinearity="relu",
            bias=bias,
            batch_first=True,
        )
        self.linear = th.nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, x, init_state=None):
        output, final_state = self.rnn(x[..., None], init_state)
        return self.linear(final_state.squeeze(0))

    @property
    def device(self):
        return self.linear.weight.device

    @property
    def dtype(self):
        return self.linear.weight.dtype


class DistRNN(OneLayerRNN):
    def __init__(self, hidden_size, seq_len, bias=False):
        self.seq_len = seq_len
        super().__init__(hidden_size, seq_len, bias=bias)


class ScalarRNN(OneLayerRNN):
    def __init__(self, hidden_size, seq_len, bias=False):
        self.seq_len = seq_len
        super().__init__(hidden_size, 1, bias=bias)


class AttentionOnlyTransformer(th.nn.Module):
    def __init__(
        self,
        input_range,
        seq_len,
        hidden_size,
        output_size,
        n_layers=1,
        n_heads=1,
        bias=False,
    ):
        self.input_range = input_range
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.bias = bias
        super().__init__()
        self.embed = th.nn.Embedding(input_range, hidden_size)
        self.pos_embed = th.nn.Embedding(seq_len, hidden_size)
        self.attns = th.nn.ModuleList(
            [
                th.nn.MultiheadAttention(
                    hidden_size, num_heads=n_heads, bias=bias, batch_first=True
                )
                for _ in range(n_layers)
            ]
        )
        self.unembed = th.nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, x):
        pos = th.arange(x.shape[-1], dtype=x.dtype, device=x.device)[None]
        state = self.embed(x) + self.pos_embed(pos)
        for attn in self.attns:
            state = state + attn(state, state, state, need_weights=False)[0]
        return self.unembed(state)[:, -1]

    @property
    def device(self):
        return self.unembed.weight.device

    @property
    def dtype(self):
        return self.unembed.weight.dtype


class DistTransformer(AttentionOnlyTransformer):
    def __init__(
        self, hidden_size, seq_len, input_range=None, n_layers=1, n_heads=1, bias=False
    ):
        if input_range is None:
            input_range = seq_len
        self.seq_len = seq_len
        super().__init__(
            input_range,
            seq_len,
            hidden_size,
            seq_len,
            n_layers=n_layers,
            n_heads=n_heads,
            bias=bias,
        )


class ScalarTransformer(AttentionOnlyTransformer):
    def __init__(
        self, hidden_size, seq_len, input_range=None, n_layers=1, n_heads=1, bias=False
    ):
        if input_range is None:
            input_range = seq_len
        self.seq_len = seq_len
        super().__init__(
            input_range,
            seq_len,
            hidden_size,
            1,
            n_layers=n_layers,
            n_heads=n_heads,
            bias=bias,
        )


class MLP(th.nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, n_hidden_layers=1, bias=False
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_hidden_layers = n_hidden_layers
        self.bias = bias
        super().__init__()
        if n_hidden_layers == 0:
            self.hidden_size = None
            self.linears = th.nn.ModuleList(
                [th.nn.Linear(input_size, output_size, bias=bias)]
            )
        else:
            self.linears = th.nn.ModuleList(
                [th.nn.Linear(input_size, hidden_size, bias=bias)]
                + [
                    th.nn.Linear(hidden_size, hidden_size, bias=bias)
                    for _ in range(n_hidden_layers - 1)
                ]
                + [th.nn.Linear(hidden_size, output_size, bias=bias)]
            )

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:
                x = th.nn.functional.relu(x)
        return x

    @property
    def device(self):
        return self.linears[-1].weight.device

    @property
    def dtype(self):
        return self.linears[-1].weight.dtype


class DistMLP(MLP):
    def __init__(self, hidden_size, seq_len, n_hidden_layers=1, bias=False):
        self.seq_len = seq_len
        super().__init__(
            seq_len, hidden_size, seq_len, n_hidden_layers=n_hidden_layers, bias=bias
        )


class ScalarMLP(MLP):
    def __init__(self, hidden_size, seq_len, n_hidden_layers=1, bias=False):
        self.seq_len = seq_len
        super().__init__(
            seq_len, hidden_size, 1, n_hidden_layers=n_hidden_layers, bias=bias
        )
