import torch as th

from .architectures import DistMLP, DistRNN


def handcrafted_2nd_argmax_2():
    model = DistRNN(hidden_size=2, seq_len=2, bias=False)
    model.rnn.weight_ih_l0.data = th.tensor(
        [[1], [-1]], dtype=model.dtype, device=model.device
    )
    model.rnn.weight_hh_l0.data = th.tensor(
        [[-1, 1], [1, -1]], dtype=model.dtype, device=model.device
    )
    model.linear.weight.data = th.tensor(
        [[1, -1], [-1, 1]], dtype=model.dtype, device=model.device
    )
    return model


def handcrafted_2nd_argmax_3():
    model = DistRNN(hidden_size=4, seq_len=3, bias=False)
    # accuracy converges to 100% as lam -> infinity, but only at high precision
    # lam = 100 works well for float32
    lam = 100
    fc1 = th.tensor(
        [[1, -lam - 1, lam], [-1, -lam, lam + 1]],
        dtype=model.dtype,
        device=model.device,
    )
    fc2 = th.tensor(
        [[1, 1], [-lam, lam + 1], [lam + 1, -lam]],
        dtype=model.dtype,
        device=model.device,
    )
    U = th.tensor(
        [[1, 0, 0, -1], [0, 1, -1, 0]], dtype=model.dtype, device=model.device
    )
    V = th.tensor([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=model.dtype, device=model.device)
    model.rnn.weight_ih_l0.data = U.T @ fc1[:, 2:]
    model.rnn.weight_hh_l0.data = U.T @ fc1[:, :2] @ th.linalg.inv(fc1[:, 1:]) @ U
    model.linear.weight.data = fc2 @ V
    return model


def handcrafted_2nd_argmax_10():
    model = DistRNN(hidden_size=22, seq_len=10, bias=False)
    with th.no_grad():
        model.rnn.weight_ih_l0.zero_()
        model.rnn.weight_hh_l0.zero_()
        model.linear.weight.zero_()
        # neurons 0-9 are max(0, x_{t-10}), ..., max(0, x_{t-1})
        model.rnn.weight_ih_l0[9] = 1.0
        for i in range(9):
            model.rnn.weight_hh_l0[i, i + 1] = 1.0
        # neuron 19 is max(0, x_0, ..., x_{t-1}) - max(0, x_0, ..., x_{t-2})
        # neuron 20 is max(0, x_0, ..., x_{t-2})
        # neuron 21 is max(0, x_0, ..., x_{t-1}) - x_{t-1}
        # neuron 18 is max(0, x_0, ..., x_{t-3}, x_{t-1}) - x_{t-1}
        model.rnn.weight_ih_l0[19] = 1.0
        model.rnn.weight_hh_l0[19, 19] = -1.0
        model.rnn.weight_hh_l0[19, 20] = -1.0
        model.rnn.weight_hh_l0[20, 19] = 1.0
        model.rnn.weight_hh_l0[20, 20] = 1.0
        model.rnn.weight_ih_l0[21] = -1.0
        model.rnn.weight_hh_l0[21, 19] = 1.0
        model.rnn.weight_hh_l0[21, 20] = 1.0
        model.rnn.weight_hh_l0[18, 20] = 1.0
        model.rnn.weight_ih_l0[18] = -1.0
        # neurons 10-17 are max(0, x_1, ..., x_{t-1}) - x_{t-1}, ..., max(0, x_0, ..., x_{t-4}, x_{t-2}, x_{t-1}) - x_{t-1}
        for i in range(10, 18):
            model.rnn.weight_hh_l0[i, i + 1] = 1.0
            model.rnn.weight_hh_l0[i, 21] = -1.0
            model.rnn.weight_hh_l0[i, 20] = 1.0
            model.rnn.weight_hh_l0[i, 19] = 1.0
            model.rnn.weight_ih_l0[i] = -1.0
        # the ith logit is max(0, x_i) - 2 * max(0, x_0, ..., x_{i-1}, x_{i+1}, ..., x_9)
        for i in range(10):
            model.linear.weight[i, i] = 1.0
        for i in range(9):
            model.linear.weight[i, 10 + i] = 1.0 * 2
            model.linear.weight[i, 21] = -1.0 * 2
            model.linear.weight[i, 20] = 1.0 * 2
            model.linear.weight[i, 19] = 1.0 * 2
        model.linear.weight[9, 20] = 1.0 * 2
    return model


def handcrafted_2nd_argmax(seq_len) -> DistRNN:
    if seq_len == 2:
        return handcrafted_2nd_argmax_2()
    elif seq_len == 3:
        return handcrafted_2nd_argmax_3()
    elif seq_len == 10:
        return handcrafted_2nd_argmax_10()
    else:
        raise ValueError(
            f"No handcrafted 2nd argmax model with seq_len {seq_len}, "
            "please choose one of: 2, 3, 10"
        )


def handcrafted_kth_argmax_mlp(seq_len, k) -> DistMLP:
    model = DistMLP(
        hidden_size=seq_len * (seq_len - 1),
        seq_len=seq_len,
        n_hidden_layers=1,
        bias=False,
    )
    with th.no_grad():
        W1 = model.linears[0].weight
        W2 = model.linears[1].weight
        W1.zero_()
        W2.zero_()
        h = 0
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    continue
                W1[h, i] = 1
                W1[h, j] = -1
                W2[i, h] = -(k - 1)
                W2[j, h] = -(seq_len - k)
                h += 1
    return model
