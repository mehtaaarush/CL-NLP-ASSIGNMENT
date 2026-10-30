"""Recomputes every numerical answer in AarushMehta_500119496.pdf.

Run with: python verify_solutions.py
PyTorch is optional; if it is installed, Q1 and Q2 are also cross-checked
against a real nn.LSTM.
"""
import math


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def r4(v):
    return round(v, 4)


def lstm_params(inputs, hidden, biases_per_gate=1):
    return 4 * (inputs * hidden + hidden * hidden + biases_per_gate * hidden)


def q1():
    x, h_prev, c_prev = 0.5, 0.2, 0.4
    z_f = 0.7 * x + 0.3 * h_prev + 0.1
    z_i = 0.6 * x + 0.4 * h_prev + 0.0
    z_c = 0.9 * x + 0.2 * h_prev + 0.0
    z_o = 0.5 * x + 0.5 * h_prev + 0.1
    # Each value is rounded to 4 d.p. before being used in the next step.
    f = r4(sigmoid(z_f))
    i = r4(sigmoid(z_i))
    c_tilde = r4(math.tanh(z_c))
    c = r4(f * c_prev + i * c_tilde)
    o = r4(sigmoid(z_o))
    tanh_c = r4(math.tanh(c))
    h = r4(o * tanh_c)
    return dict(z_f=r4(z_f), z_i=r4(z_i), z_c=r4(z_c), z_o=r4(z_o), f=f, i=i,
                c_tilde=c_tilde, c=c, o=o, tanh_c=tanh_c, h=h)


def q2():
    return dict(single=lstm_params(10, 32), double=lstm_params(10, 32, 2))


def q3():
    return dict(a10=0.9 ** 10, a50=0.9 ** 50, b_05=0.5 ** 50, b_099=0.99 ** 50)


def q4():
    records = 3 * 365 * 24
    lookback, horizon = 168, 24
    windows = records - lookback - horizon + 1
    train_records = int(records * 7 / 10)
    train_windows = train_records - lookback - horizon + 1
    return dict(records=records, windows=windows, train_records=train_records,
                train_windows=train_windows,
                batches=math.ceil(train_windows / 64),
                full_batches=train_windows // 64,
                last_batch=train_windows % 64,
                params=lstm_params(3, 64))


def q5():
    total, positives, flagged, tp = 1000, 40, 45, 30
    fp = flagged - tp
    fn = positives - tp
    tn = total - tp - fp - fn
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return dict(norm=(95 - 20) / (120 - 20), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall),
                accuracy=(tp + tn) / total,
                always_ok=(total - positives) / total)


def torch_checks():
    """Returns (C_t, h_t) from nn.LSTM with the Q1 weights, and Q2 param count."""
    import torch

    cell = torch.nn.LSTM(1, 1)
    # PyTorch orders the gates: input, forget, candidate, output.
    state = {
        "weight_ih_l0": [[0.6], [0.7], [0.9], [0.5]],
        "weight_hh_l0": [[0.4], [0.3], [0.2], [0.5]],
        "bias_ih_l0": [0.0, 0.1, 0.0, 0.1],
        "bias_hh_l0": [0.0, 0.0, 0.0, 0.0],
    }
    with torch.no_grad():
        for name, value in state.items():
            getattr(cell, name).copy_(torch.tensor(value, dtype=torch.float64).float())
        _, (h, c) = cell(torch.tensor([[[0.5]]]),
                         (torch.tensor([[[0.2]]]), torch.tensor([[[0.4]]])))
    count = sum(p.numel() for p in torch.nn.LSTM(10, 32).parameters())
    return c.item(), h.item(), count


if __name__ == "__main__":
    for name, solve in (("Q1", q1), ("Q2", q2), ("Q3", q3), ("Q4", q4), ("Q5", q5)):
        print(name, solve())

    try:
        c, h, count = torch_checks()
    except ImportError:
        print("PyTorch not installed; skipping nn.LSTM cross-check")
    else:
        print(f"nn.LSTM check: C_t={c:.4f} h_t={h:.4f} params={count}")
        assert abs(c - q1()["c"]) < 2e-4 and abs(h - q1()["h"]) < 2e-4
        assert count == q2()["double"]
        print("Cross-check passed")
