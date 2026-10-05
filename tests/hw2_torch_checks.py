"""Shared public and upload checks for the PyTorch assignments.

Expected numerical results come from fixtures computed independently of the
submission. Each named check runs separately so feedback survives other errors.
"""

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


def _numpy(value):
    return np.asarray(value, dtype=np.float64)


def _tensor(suite, value, shape, what, dtype=None):
    suite.require(isinstance(value, torch.Tensor), f"{what} must be a torch.Tensor.")
    suite.require(tuple(value.shape) == tuple(shape),
                  f"{what} has shape {tuple(value.shape)}; expected {tuple(shape)}.")
    if dtype is not None:
        suite.require(value.dtype == dtype, f"{what} must have dtype {dtype}; got {value.dtype}.")
    suite.require(bool(torch.isfinite(value).all()), f"{what} contains NaN or infinity.")
    return value.detach().cpu().numpy()


def _copy_parameters(model, case):
    with torch.no_grad():
        for parameter, key in ((model.fc1.weight, "W1"), (model.fc1.bias, "b1"),
                               (model.fc2.weight, "W2"), (model.fc2.bias, "b2")):
            parameter.copy_(torch.as_tensor(case[key], dtype=parameter.dtype))


def dataset(suite, hw, cases):
    for seed in cases["seeds"]:
        def check(seed=seed):
            n = 4096
            before = torch.random.get_rng_state().clone()
            ds = hw.NoisyXOR(n=n, noise=0, seed=seed)
            suite.require(torch.equal(before, torch.random.get_rng_state()),
                          "Dataset creation changed the global PyTorch RNG. Pass a local seeded generator to random operations.")
            X = _tensor(suite, ds.X, (n, 2), "dataset.X", torch.float32)
            y = _tensor(suite, ds.y, (n, 1), "dataset.y", torch.float32)
            suite.require(len(ds) == n, "Dataset length must equal the requested n.")
            suite.require(bool(np.isin(X, [0, 1]).all()), "With noise=0, every input must be an exact corner.")
            expected = np.logical_xor(X[:, 0], X[:, 1]).astype(np.float32)[:, None]
            suite.close(y, expected, "XOR labels at noise=0")
            corner_ids = X[:, 0].astype(int) * 2 + X[:, 1].astype(int)
            counts = np.bincount(corner_ids, minlength=4)
            suite.require(bool(np.all(np.abs(counts / n - .25) < .04)),
                          f"All four corners must be sampled uniformly; observed counts {counts.tolist()}.")
            suite.require(abs(float(y.mean()) - .5) < .06,
                          f"Classes should be balanced; observed mean label {y.mean():.3f}. Check torch.randint's exclusive upper bound.")
            for i in (0, 37, n - 1):
                sample = ds[i]
                suite.require(isinstance(sample, (tuple, list)) and len(sample) == 2,
                              "dataset[i] must return an (x, y) pair.")
                suite.close(_tensor(suite, sample[0], (2,), "sample input"), X[i], "sample input")
                suite.close(_tensor(suite, sample[1], (1,), "sample label"), y[i], "sample label")
        suite.run(f"Zero noise, balanced corners and labels (seed {seed})", check)

        for noise in (.05, .2):
            def check(seed=seed, noise=noise):
                before = torch.random.get_rng_state().clone()
                ds = hw.NoisyXOR(n=4096, noise=noise, seed=seed)
                repeated = hw.NoisyXOR(n=4096, noise=noise, seed=seed)
                suite.require(torch.equal(before, torch.random.get_rng_state()),
                              "Use a local torch.Generator; dataset creation must not change the global RNG.")
                X = _tensor(suite, ds.X, (4096, 2), "dataset.X", torch.float32)
                y = _tensor(suite, ds.y, (4096, 1), "dataset.y", torch.float32)
                suite.require(torch.equal(ds.X, repeated.X) and torch.equal(ds.y, repeated.y),
                              "Creating the dataset twice with the same seed must give identical inputs and labels.")
                different = hw.NoisyXOR(n=4096, noise=noise, seed=seed + 1)
                suite.require(not torch.equal(ds.X, different.X),
                              "Different seeds must produce different datasets; use the supplied seed rather than a fixed constant.")
                suite.require(bool(np.isin(y, [0, 1]).all()), "Labels must be binary, even when inputs are noisy.")
                corners = np.clip(np.round(X), 0, 1)
                residual = X - corners
                std = residual.std(axis=0)
                suite.require(bool(np.all(np.abs(std - noise) < noise * .15)),
                              f"noise is the standard deviation: requested {noise:.4f}, observed {std.tolist()}. Scale Gaussian noise by noise, not noise**2.")
                suite.require(bool(np.all(np.abs(residual.mean(axis=0)) < noise * .07)),
                              "Gaussian noise should be centered at zero in both coordinates.")
                correlation = np.corrcoef(residual.T)[0, 1]
                suite.require(abs(float(correlation)) < .08,
                              "The Gaussian noise in the two coordinates must be independent.")
                if noise == .05:
                    tail_fraction = float(np.mean(np.abs(residual) > 2 * noise))
                    suite.require(.025 < tail_fraction < .075,
                                  f"Noise should be Gaussian, not uniform or constant; fraction beyond two standard deviations is {tail_fraction:.3f}.")
                recovered = np.logical_xor(corners[:, 0], corners[:, 1])
                mismatches = float(np.mean(recovered != y[:, 0]))
                suite.require(mismatches < (.001 if noise == .05 else .05),
                              f"Labels should belong to the sampled corner before adding noise; nearest-corner mismatch rate is {mismatches:.3f}.")
            suite.run(f"Noise standard deviation and reproducibility ({noise}, seed {seed})", check)


def loaders(suite, hw, cases):
    for seed in cases["seeds"]:
        for n, fraction, batch_size in ((11, .3, 3), (53, .17, 8)):
            def check(seed=seed, n=n, fraction=fraction, batch_size=batch_size):
                ids = torch.arange(n, dtype=torch.float32)
                X = torch.stack((ids, ids + .125), dim=1)
                y = (n - ids).reshape(n, 1)
                ds = TensorDataset(X, y)
                train_loader, val_loader = hw.make_loaders(ds, val_fraction=fraction,
                                                           batch_size=batch_size, seed=seed)
                suite.require(isinstance(train_loader, DataLoader) and isinstance(val_loader, DataLoader),
                              "make_loaders must return two PyTorch DataLoaders.")
                n_val = int(n * fraction)
                suite.require(len(train_loader.dataset) == n - n_val and len(val_loader.dataset) == n_val,
                              f"Use n_val=int(n*val_fraction): expected split sizes {n-n_val} and {n_val}.")
                expected = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
                suite.require(list(train_loader.dataset.indices) == expected[:n-n_val] and
                              list(val_loader.dataset.indices) == expected[n-n_val:],
                              "Split indices must match random_split with a local generator seeded by seed.")
                suite.require(train_loader.batch_size == batch_size and val_loader.batch_size == batch_size,
                              "Both loaders must use the requested batch_size.")
                suite.require(isinstance(train_loader.sampler, RandomSampler),
                              "The training DataLoader must shuffle every epoch (shuffle=True).")
                suite.require(isinstance(val_loader.sampler, SequentialSampler),
                              "The validation DataLoader must preserve its split order (shuffle=False).")
                def collect(loader):
                    visited = []
                    for xb, yb in loader:
                        suite.require(0 < len(xb) <= batch_size, "Incorrect batch size.")
                        suite.close(xb[:, 1].numpy(), xb[:, 0].numpy() + .125, "Input coordinates stay paired")
                        suite.close(yb[:, 0].numpy(), n - xb[:, 0].numpy(), "Labels stay paired with inputs")
                        visited.extend(xb[:, 0].to(torch.int64).tolist())
                    return visited
                orders = [collect(train_loader) for _ in range(3)]
                for order in orders:
                    suite.require(sorted(order) == sorted(expected[:n-n_val]),
                                  "Each training epoch must visit every training sample exactly once, including the last partial batch.")
                suite.require(any(order != orders[0] for order in orders[1:]),
                              "Training order repeated across three epochs; make sure shuffling runs each epoch.")
                for _ in range(2):
                    suite.require(collect(val_loader) == expected[n-n_val:],
                                  "Validation order must stay fixed and include every validation sample.")
            suite.run(f"Split, pairing and batching (n={n}, seed {seed})", check)


def model(suite, hw, cases):
    activations = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}
    for index, case in enumerate(cases["networks"]):
        def check(case=case):
            W1, W2 = _numpy(case["W1"]), _numpy(case["W2"])
            mlp = hw.TorchMLP(W1.shape[1], W1.shape[0], W2.shape[0], activation=case["activation"])
            suite.require(isinstance(mlp, nn.Module), "TorchMLP must inherit from nn.Module.")
            suite.require(isinstance(mlp.fc1, nn.Linear) and isinstance(mlp.fc2, nn.Linear),
                          "fc1 and fc2 must be registered nn.Linear layers.")
            suite.require(isinstance(mlp.act, activations[case["activation"]]),
                          f"The requested hidden activation is {case['activation']}.")
            suite.require(tuple(mlp.fc1.weight.shape) == W1.shape and tuple(mlp.fc2.weight.shape) == W2.shape,
                          "Linear layer dimensions do not match in_dim, hidden_dim and out_dim.")
            registered = {id(parameter) for parameter in mlp.parameters()}
            suite.require(all(id(parameter) in registered for parameter in
                              (mlp.fc1.weight, mlp.fc1.bias, mlp.fc2.weight, mlp.fc2.bias)),
                          "All four weights/biases must be registered parameters for SGD to update them.")
            mlp.double()
            _copy_parameters(mlp, case)
            probes = case.get("probes", [case["cache"]])
            X = np.stack([_numpy(probe["x"]) for probe in probes])
            expected = np.stack([_numpy(probe["z2"]) for probe in probes])
            for count in (1, len(X)):
                output = mlp(torch.from_numpy(X[:count]))
                actual = _tensor(suite, output, (count, W2.shape[0]), "TorchMLP output", torch.float64)
                suite.close(actual, expected[:count], "Forward logits (do not apply a final sigmoid)", atol=1e-10, rtol=1e-10)
        suite.run(f"{case['activation']} forward, dimensions and parameters (case {index+1})", check)


def bridge(suite, hw, cases):
    for index, case in enumerate(cases["networks"]):
        def check(case=case):
            # Deliberately independent of the student's NumPy MLP implementation.
            arrays = {key: _numpy(case[key]).copy() for key in ("W1", "b1", "W2", "b2")}
            np_model = SimpleNamespace(fc1=SimpleNamespace(W=arrays["W1"], b=arrays["b1"]),
                                       fc2=SimpleNamespace(W=arrays["W2"], b=arrays["b2"]))
            before = {key: value.copy() for key, value in arrays.items()}
            x, y = _numpy(case["x"]), _numpy(case["y"])
            x_before, y_before = x.copy(), y.copy()
            gradients = hw.torch_gradients(np_model, x, y, activation=case["activation"])
            suite.require(isinstance(gradients, dict), "torch_gradients must return a dictionary.")
            for name in ("fc1.W", "fc1.b", "fc2.W", "fc2.b"):
                suite.require(name in gradients, f"Gradient dictionary is missing {name!r}.")
                expected = _numpy(case["bce_gradients"][name])
                suite.array(gradients[name], expected.shape, f"grads[{name!r}]")
                suite.require(gradients[name].dtype == np.float64,
                              f"grads[{name!r}] must be float64; convert the Torch model and inputs with .double().")
                suite.close(gradients[name], expected, f"grads[{name!r}] for {case['activation']}", atol=1e-10, rtol=1e-9)
            for key, value in arrays.items():
                suite.close(value, before[key], f"Original NumPy parameter {key} remains unchanged", atol=0, rtol=0)
            suite.close(x, x_before, "Original input remains unchanged", atol=0, rtol=0)
            suite.close(y, y_before, "Original target remains unchanged", atol=0, rtol=0)
        suite.run(f"Autograd bridge with explicit {case['activation']} (case {index+1})", check)


class _EvaluationProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x):
        self.calls.append((self.training, torch.is_grad_enabled(), len(x)))
        return x[:, :1]


def evaluate(suite, hw, cases):
    for batch_size in (1, 5, 20):
        def check(batch_size=batch_size):
            logits = torch.tensor([-3., 0., .2, 8., -.4, 0., 2., -1., .1, -.1, 0., 4., -8.])[:, None]
            y = torch.tensor([0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 1., 1., 0.])[:, None]
            loader = DataLoader(TensorDataset(logits, y), batch_size=batch_size, shuffle=False)
            probe = _EvaluationProbe()
            probe.train()
            accuracy = hw.evaluate(probe, loader)
            expected = float(((logits > 0) == y.bool()).float().sum().item() / len(y))
            suite.require(type(accuracy) is float, "evaluate must return a Python float.")
            suite.close(np.asarray(accuracy), np.asarray(expected),
                        "Accuracy over all samples; class 1 iff logit > 0 (zero is class 0)")
            suite.require(sum(call[2] for call in probe.calls) == len(y), "Evaluate every sample exactly once.")
            suite.require(all(not call[0] for call in probe.calls), "Call model.eval() before evaluation.")
            suite.require(all(not call[1] for call in probe.calls), "Evaluate inside torch.no_grad().")
        suite.run(f"Accuracy, zero logits and evaluation mode (batch_size={batch_size})", check)
    for batch_size in (1, 3, 20):
        def check(batch_size=batch_size):
            # Every label agrees with the documented boundary, preventing two
            # wrong classifications from cancelling in the aggregate accuracy.
            logits = torch.tensor([.01, .2, .49, 0., -.01])[:, None]
            y = torch.tensor([1., 1., 1., 0., 0.])[:, None]
            loader = DataLoader(TensorDataset(logits, y), batch_size=batch_size, shuffle=False)
            probe = _EvaluationProbe()
            probe.train()
            accuracy = hw.evaluate(probe, loader)
            suite.require(type(accuracy) is float, "evaluate must return a Python float.")
            suite.close(np.asarray(accuracy), np.asarray(1.),
                        "All boundary examples should be correct: positive logits predict 1, zero and negative logits predict 0")
            suite.require(sum(call[2] for call in probe.calls) == len(y), "Evaluate every sample exactly once.")
            suite.require(all(not call[0] for call in probe.calls), "Call model.eval() before evaluation.")
            suite.require(all(not call[1] for call in probe.calls), "Evaluate inside torch.no_grad().")
        suite.run(f"Strict logit boundary at zero (batch_size={batch_size})", check)


class _TrainingProbe(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)
        self.training_calls = []
        self.training_logits = []

    def forward(self, x):
        output = super().forward(x)
        if torch.is_grad_enabled():
            self.training_calls.append(self.training)
            self.training_logits.append(output.detach().clone())
        return output


def train(suite, hw, cases):
    for index, case in enumerate(cases["torch_training"]):
        def check(case=case, index=index):
            network = case["network"]
            W1, W2 = _numpy(network["W1"]), _numpy(network["W2"])
            activation = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}[network["activation"]]
            mlp = _TrainingProbe(nn.Linear(W1.shape[1], W1.shape[0]), activation(),
                                 nn.Linear(W2.shape[1], W2.shape[0]))
            with torch.no_grad():
                for parameter, key in zip(mlp.parameters(), ("W1", "b1", "W2", "b2")):
                    parameter.copy_(torch.tensor(network[key], dtype=torch.float32))
                    # zero_grad must clear existing gradients before the first update.
                    parameter.grad = torch.ones_like(parameter)
            mlp.eval()
            stale_gradients = []
            for parameter in mlp.parameters():
                parameter.register_hook(lambda gradient, parameter=parameter:
                                        stale_gradients.append(parameter.grad is not None and
                                                               bool(parameter.grad.abs().sum() > 0)))
            ds = TensorDataset(torch.tensor(case["X"], dtype=torch.float32),
                               torch.tensor(case["y"], dtype=torch.float32))
            loader = DataLoader(ds, batch_size=case["batch_size"], shuffle=False)
            validation_calls = []
            def controlled_evaluate(model, val_loader):
                validation_calls.append((model, val_loader))
                model.eval()
                with torch.no_grad():
                    correct, count = 0, 0
                    for xb, yb in val_loader:
                        correct += int(((model(xb) > 0) == yb.bool()).sum().item())
                        count += len(xb)
                    return float(correct / count)
            original = hw.evaluate
            hw.evaluate = controlled_evaluate
            try:
                history = hw.train_torch(mlp, loader, loader, epochs=case["epochs"], lr=case["lr"])
            finally:
                hw.evaluate = original
            suffix = f" ({network['activation']}, case {index+1})"
            suite.run("Clear gradients before backward" + suffix,
                      lambda: suite.require(stale_gradients and not any(stale_gradients),
                                            "Old gradients were present when backward started. Clear gradients before "
                                            "the first batch and every subsequent batch; clearing only after step() "
                                            "leaves pre-existing gradients in the first update."))

            def modes():
                suite.require(len(mlp.training_calls) == len(loader) * case["epochs"],
                              "Run a training forward pass once per batch in every epoch.")
                suite.require(all(mlp.training_calls),
                              "Call model.train() at the start of every epoch; evaluate switches it to eval mode.")
                suite.require(len(validation_calls) == case["epochs"], "Call evaluate(model, val_loader) after every epoch.")
                suite.require(all(model is mlp and val_loader is loader for model, val_loader in validation_calls),
                              "Evaluate the trained model on the supplied validation loader.")
            suite.run("Training and evaluation modes" + suffix, modes)

            def structure():
                suite.require(isinstance(history, dict), "train_torch must return a history dictionary.")
                for key in ("train_loss", "val_acc"):
                    suite.require(key in history and isinstance(history[key], list) and len(history[key]) == case["epochs"],
                                  f"history[{key!r}] must be a list with one entry per epoch.")
            suite.run("History structure" + suffix, structure)

            if isinstance(history, dict) and isinstance(history.get("train_loss"), list):
                def averaging():
                    suite.require(len(mlp.training_logits) == len(loader) * case["epochs"],
                                  "Cannot verify epoch means without one recorded forward pass per batch.")
                    expected_means = []
                    for epoch in range(case["epochs"]):
                        total = 0.
                        for batch in range(len(loader)):
                            start = batch * case["batch_size"]
                            targets = ds.tensors[1][start:start + case["batch_size"]]
                            logits = mlp.training_logits[epoch * len(loader) + batch]
                            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets).item()
                            total += loss * len(targets)
                        expected_means.append(total / len(ds))
                    # Use the observed pre-update predictions: this isolates averaging
                    # from a separate mistake in the SGD updates.
                    suite.close(history["train_loss"], expected_means,
                                "Epoch train_loss must average over samples, not batches. "
                                "The final batch is smaller: weight each batch mean by its actual sample count",
                                atol=2e-6, rtol=2e-6)
                suite.run("Sample-weighted epoch loss" + suffix, averaging)

            def reference_trace():
                structure()
                for key in ("train_loss", "val_acc"):
                    suite.close(history[key], case["history"][key], f"Reference {key} after sequential SGD",
                                atol=2e-6, rtol=2e-6)
                for parameter, key in zip(mlp.parameters(), ("W1", "b1", "W2", "b2")):
                    suite.close(parameter.detach().numpy(), _numpy(case["parameters"][key]),
                                f"Final parameter {key}; clear gradients before each batch and use the requested SGD learning rate",
                                atol=2e-6, rtol=2e-6)
            suite.run("Reference SGD history and final parameters" + suffix, reference_trace)
        suite.run(f"Deterministic SGD, partial batches and histories (case {index+1})", check)
