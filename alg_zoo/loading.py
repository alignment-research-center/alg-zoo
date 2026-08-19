import os

import blobfile as bf
import torch as th

from .architectures import (
    DistMLP,
    DistRNN,
    DistTransformer,
    ScalarMLP,
    ScalarRNN,
    ScalarTransformer,
)


def load_state_dict(model_path):
    prev_auth = os.getenv("BLOBFILE_FORCE_GOOGLE_ANONYMOUS_AUTH")
    try:
        os.environ["BLOBFILE_FORCE_GOOGLE_ANONYMOUS_AUTH"] = "1"
        with bf.BlobFile(model_path, "rb") as model_file:
            state_dict = th.load(model_file, weights_only=True, map_location="cpu")
        return state_dict
    finally:
        if prev_auth is None:
            os.environ.pop("BLOBFILE_FORCE_GOOGLE_ANONYMOUS_AUTH", None)
        else:
            os.environ["BLOBFILE_FORCE_GOOGLE_ANONYMOUS_AUTH"] = prev_auth


def load_dist_rnn(model_path, bias=False):
    state_dict = load_state_dict(model_path)
    seq_len, hidden_size = state_dict["linear.weight"].shape
    model = DistRNN(hidden_size=hidden_size, seq_len=seq_len, bias=bias)
    model.load_state_dict(state_dict)
    return model


def load_scalar_rnn(model_path, seq_len, bias=False):
    state_dict = load_state_dict(model_path)
    _, hidden_size = state_dict["linear.weight"].shape
    model = ScalarRNN(hidden_size=hidden_size, seq_len=seq_len, bias=bias)
    model.load_state_dict(state_dict)
    return model


def load_transformer(model_cls, model_path, n_heads=1, bias=False):
    state_dict = load_state_dict(model_path)
    input_range, hidden_size = state_dict["embed.weight"].shape
    seq_len = state_dict["pos_embed.weight"].shape[0]
    attn_indices = [
        key.split(".")[1] for key in state_dict.keys() if key.split(".")[0] == "attns"
    ]
    n_layers = len(set(attn_indices))
    model = model_cls(
        hidden_size=hidden_size,
        seq_len=seq_len,
        input_range=input_range,
        n_layers=n_layers,
        n_heads=n_heads,
        bias=bias,
    )
    model.load_state_dict(state_dict)
    return model


def load_dist_transformer(model_path, **kwargs):
    return load_transformer(DistTransformer, model_path, **kwargs)


def load_scalar_transformer(model_path, **kwargs):
    return load_transformer(ScalarTransformer, model_path, **kwargs)


def load_mlp(model_cls, model_path, bias=False):
    state_dict = load_state_dict(model_path)
    linear_indices = [
        key.split(".")[1] for key in state_dict.keys() if key.split(".")[0] == "linears"
    ]
    n_hidden_layers = len(set(linear_indices)) - 1
    if n_hidden_layers == 0:
        _, seq_len = state_dict["linears.0.weight"].shape
        hidden_size = None
    else:
        hidden_size, seq_len = state_dict["linears.0.weight"].shape
    model = model_cls(
        hidden_size=hidden_size,
        seq_len=seq_len,
        n_hidden_layers=n_hidden_layers,
        bias=bias,
    )
    model.load_state_dict(state_dict)
    return model


def load_dist_mlp(model_path, **kwargs):
    return load_mlp(DistMLP, model_path, **kwargs)


def load_scalar_mlp(model_path, **kwargs):
    return load_mlp(ScalarMLP, model_path, **kwargs)


def example_2nd_argmax() -> DistRNN:
    # 2nd argmax model with hidden_size=16, seq_len=10 discussed in blog post
    return load_dist_rnn(
        "gs://arc-ml-public/alg/one_layer_16_hidden_94_acc_2nd_argmax.pth",
        bias=False,
    )
