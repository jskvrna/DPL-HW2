"""Shared HW2 checks. BRUTE runs this same code with additional seeded fixtures.

Expected network values live in hw2_cases.json; no student implementation is
used to generate them. This module does not import any autodiff library.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import traceback
import warnings

import numpy as np

from hw2_policy import ForbiddenImport


@dataclass(frozen=True)
class Task:
    key: str
    tid: str
    title: str
    points: float


TASKS = (
    Task("1.1.loader", "1.1", "XORDataLoader", .5),
    Task("1.2.init", "1.2", "Linear initialization", .5),
    Task("1.3.sigmoid", "1.3", "Sigmoid", .5),
    Task("1.3.tanh", "1.3", "Tanh", .5),
    Task("1.3.relu", "1.3", "ReLU", .5),
    Task("1.4.linear", "1.4", "Linear forward / backward", 1.),
    Task("1.4.bce", "1.4", "BCELoss forward / backward", .5),
    Task("1.5.numgrad", "1.5", "numerical_grad", .5),
    Task("1.6.forward", "1.6", "MLP.forward", .5),
    Task("1.6.backward", "1.6", "MLP.backward", 1.),
    Task("1.7.sgd", "1.7", "sgd_step", .25),
    Task("1.7.train", "1.7", "train", .75),
    Task("2.1.dataset", "2.1", "NoisyXOR dataset", .25),
    Task("2.1.loaders", "2.1", "make_loaders", .25),
    Task("2.2.model", "2.2", "TorchMLP", .5),
    Task("2.3.bridge", "2.3", "torch_gradients (bridge)", 1.),
    Task("2.4.evaluate", "2.4", "evaluate", .25),
    Task("2.4.train", "2.4", "train_torch", .75),
)


def select_tasks(selectors=None):
    if not selectors:
        return list(TASKS)
    return [task for task in TASKS if any(
        task.key == s or task.tid == s or task.tid.startswith(s.rstrip(".") + ".")
        for s in selectors)]


class Suite:
    def __init__(self):
        self.cases = []

    @staticmethod
    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    def array(self, value, shape, what):
        self.require(isinstance(value, np.ndarray), f"{what}: expected a NumPy array, got {type(value).__name__}.")
        self.require(value.shape == tuple(shape), f"{what}: expected shape {tuple(shape)}, got {value.shape}.")
        self.require(np.issubdtype(value.dtype, np.floating), f"{what}: expected floating values, got {value.dtype}.")
        self.require(bool(np.isfinite(value).all()), f"{what}: contains NaN or infinity.")

    def close(self, actual, expected, what, atol=1e-7, rtol=1e-6):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.require(actual.shape == expected.shape,
                     f"{what}: expected shape {expected.shape}, got {actual.shape}.")
        self.require(bool(np.isfinite(actual).all()), f"{what}: contains NaN or infinity.")
        if not np.allclose(actual, expected, atol=atol, rtol=rtol):
            difference = np.abs(actual - expected)
            error = float(difference.max())
            excerpt = lambda a: np.array2string(a.ravel()[:8], precision=6, separator=", ")
            raise AssertionError(f"{what}: values differ (maximum absolute error {error:.6g}; "
                                 f"atol={atol:g}, rtol={rtol:g}).\n"
                                 f"Expected excerpt: {excerpt(expected)}\nActual excerpt:   {excerpt(actual)}")

    def run(self, name, function):
        case = {"name": name, "status": "PASSED", "message": "Check passed."}
        try:
            function()
        except ForbiddenImport as error:
            case.update(status="POLICY ERROR", message=str(error))
        except NotImplementedError as error:
            case.update(status="NOT IMPLEMENTED", message=f"Implementation needed: {error}.")
        except AssertionError as error:
            case.update(status="FAILED", message=str(error))
        except (Exception, SystemExit) as error:
            frames = traceback.extract_tb(error.__traceback__)
            student_frames = [frame for frame in frames if Path(frame.filename).name in
                              {"hw2_numpy.py", "hw2_torch.py"}]
            detail = "\n".join(f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}\n  {frame.line or ''}"
                               for frame in student_frames)
            case.update(status="ERROR", message=f"{type(error).__name__}: {error}", detail=detail)
        self.cases.append(case)

    def status(self):
        statuses = {case["status"] for case in self.cases}
        for status in ("POLICY ERROR", "ERROR", "FAILED", "NOT IMPLEMENTED"):
            if status in statuses:
                return status
        return "PASSED" if self.cases else "ERROR"


def _arrays(case):
    return {key: np.array(value, dtype=np.float64) if isinstance(value, list) else value
            for key, value in case.items()}


def _layer(hw, W, b):
    layer = hw.Linear.__new__(hw.Linear)
    layer.W, layer.b = np.array(W, float), np.array(b, float)
    layer.out_dim, layer.in_dim = layer.W.shape
    layer.dW = layer.db = None
    return layer


def _network(hw, raw):
    model = hw.MLP.__new__(hw.MLP)
    model.activation = raw["activation"]
    model.fc1, model.fc2 = _layer(hw, raw["W1"], raw["b1"]), _layer(hw, raw["W2"], raw["b2"])
    model.act1 = getattr(hw, {"sigmoid": "Sigmoid", "tanh": "Tanh", "relu": "ReLU"}[model.activation])()
    model.act2 = hw.Sigmoid()
    return model


def loader(suite, hw, cases):
    for seed in cases["seeds"]:
        for n, in_dim, out_dim in ((1, 2, 1), (11, 3, 2), (23, 7, 4)):
            def check(seed=seed, n=n, in_dim=in_dim, out_dim=out_dim):
                rng = np.random.default_rng(seed)
                X, y = rng.normal(size=(n, in_dim)), rng.normal(size=(n, out_dim))
                before = X.copy(), y.copy()
                for shuffle in (False, True):
                    student_rng, expected_rng = np.random.default_rng(seed), np.random.default_rng(seed)
                    data = hw.XORDataLoader(X, y, shuffle=shuffle, rng=student_rng)
                    suite.require(len(data) == n, f"Expected {n} samples, got {len(data)}.")
                    for epoch in range(4):
                        order = expected_rng.permutation(n) if shuffle else np.arange(n)
                        pairs = list(data)
                        suite.require(len(pairs) == n, f"Epoch {epoch}: expected {n} samples, got {len(pairs)}.")
                        for (x, target), index in zip(pairs, order):
                            suite.array(x, (in_dim,), "Loader input")
                            suite.array(target, (out_dim,), "Loader target")
                            suite.require(np.array_equal(x, X[index]) and np.array_equal(target, y[index]),
                                          f"Epoch {epoch}, shuffle={shuffle}: incorrect order or input/label pairing. "
                                          "Call the supplied rng.permutation once at each epoch; yield matching rows.")
                    suite.require(np.array_equal(student_rng.random(5), expected_rng.random(5)),
                                  "The loader used a different number of RNG calls than specified.")
                suite.require(np.array_equal(X, before[0]) and np.array_equal(y, before[1]),
                              "The loader modified the source dataset.")
            suite.run(f"Dataset ({n}, {in_dim}), labels ({n}, {out_dim}), repeated epochs, seed {seed}", check)


def initialization(suite, hw, cases):
    for seed in cases["seeds"]:
        def check(seed=seed):
            state = np.random.get_state()
            student_rng, expected_rng = np.random.default_rng(seed), np.random.default_rng(seed)
            for in_dim, out_dim in ((2, 3), (7, 1), (3, 8), (400, 600)):
                layer = hw.Linear(in_dim, out_dim, rng=student_rng)
                expected = expected_rng.normal(0, np.sqrt(2 / (in_dim + out_dim)), (out_dim, in_dim))
                suite.array(layer.W, (out_dim, in_dim), "Linear.W")
                suite.array(layer.b, (out_dim,), "Linear.b")
                suite.close(layer.W, expected, "Xavier normal weights; use exactly one call to the supplied RNG",
                            atol=1e-12, rtol=1e-12)
                suite.close(layer.b, np.zeros(out_dim), "Initial biases", atol=0, rtol=0)
            now = np.random.get_state()
            suite.require(all(np.array_equal(a, b) for a, b in zip(state, now)),
                          "Initialization changed the global NumPy RNG; use only the supplied rng.")
            suite.require(np.array_equal(student_rng.random(5), expected_rng.random(5)),
                          "Initialization must draw weights once per layer, without extra RNG calls.")
        suite.run(f"Xavier values, shapes, zero biases and RNG state, seed {seed}", check)


def activation(suite, hw, cases, name):
    cls = getattr(hw, name.capitalize() if name != "relu" else "ReLU")
    for index, raw in enumerate(c for c in cases["activations"] if c["name"] == name):
        def check(raw=raw):
            z, want = np.array(raw["z"], float), np.array(raw["forward"], float)
            original = z.copy()
            act = cls()
            with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                warnings.simplefilter("error", RuntimeWarning)
                value, derivative = act.forward(z), act.backward(z)
            suite.array(value, z.shape, f"{cls.__name__}.forward")
            suite.array(derivative, z.shape, f"{cls.__name__}.backward")
            suite.close(value, want, "Activation values")
            suite.close(derivative, raw["backward"], "Local activation derivative")
            suite.require(np.array_equal(z, original), "Activation modified its input.")
        suite.run(f"Element-wise values and derivatives, case {index + 1}", check)

    def edge():
        z = np.array([-1000., -50., -1e-12, -0., 0., 1e-12, 50., 1000.])
        if name == "sigmoid":
            expected = np.array([0., 0., .5 - 2.5e-13, .5, .5, .5 + 2.5e-13, 1., 1.])
            derivative = expected * (1 - expected)
        elif name == "tanh":
            expected, derivative = np.tanh(z), 1 - np.tanh(z) ** 2
        else:
            expected, derivative = np.maximum(z, 0), (z > 0).astype(float)
        with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            warnings.simplefilter("error", RuntimeWarning)
            act = cls()
            suite.close(act.forward(z), expected, "Activation at extreme values and zero", atol=1e-12)
            suite.close(act.backward(z), derivative, "Derivative at extreme values and zero", atol=1e-12)
    suite.run("Large positive/negative values and the derivative at zero", edge)


def linear(suite, hw, cases):
    for index, raw in enumerate(cases["linear"]):
        def check(raw=raw):
            c = _arrays(raw)
            layer = _layer(hw, c["W"], c["b"])
            x, upstream = c["x"].copy(), c["upstream"].copy()
            suite.close(layer.forward(x), c["z"], "Linear.forward")
            result = layer.backward(x, upstream)
            suite.require(isinstance(result, tuple) and len(result) == 3,
                          "Linear.backward must return (dL_dx, dL_dW, dL_db).")
            for value, key in zip(result, ("dx", "dW", "db")):
                suite.array(value, c[key].shape, key)
                suite.close(value, c[key], key)
            for actual, original, what in ((x, c["x"], "input"), (upstream, c["upstream"], "upstream gradient"),
                                          (layer.W, c["W"], "weights"), (layer.b, c["b"], "biases")):
                suite.require(np.array_equal(actual, original), f"Linear modified its {what} during forward/backward.")
        suite.run(f"Rectangular linear layer, arbitrary upstream gradient, case {index + 1}", check)


def bce(suite, hw, cases):
    for index, raw in enumerate(cases["bce"]):
        def check(raw=raw):
            c = _arrays(raw)
            loss_fn = hw.BCELoss()
            loss = loss_fn.forward(c["p"].copy(), c["y"].copy())
            gradient = loss_fn.backward(c["p"].copy(), c["y"].copy())
            suite.require(isinstance(loss, float), f"BCELoss.forward must return a Python float, got {type(loss).__name__}.")
            suite.close(loss, c["loss"], "Mean BCE (clip probabilities in both forward and backward)")
            suite.array(gradient, c["p"].shape, "BCE gradient")
            suite.close(gradient, c["gradient"], "BCE gradient; include division by the number of outputs")
        suite.run(f"BCE mean reduction and probability clipping, case {index + 1}", check)


def numerical(suite, hw, cases):
    for seed in cases["seeds"]:
        def check(seed=seed):
            rng = np.random.default_rng(seed)
            for shape in ((7,), (3, 4), (2, 2, 3)):
                param, coefficient = rng.normal(size=shape), rng.normal(size=shape)
                original = param.copy()
                grad = hw.numerical_grad(lambda: float(np.sum(np.sin(param) * coefficient)), param)
                suite.array(grad, shape, "numerical_grad")
                suite.close(grad, np.cos(original) * coefficient, "Numerical derivative of a trigonometric function", atol=1e-6)
                suite.require(np.array_equal(param, original), "Restore each parameter exactly after perturbation.")
                suite.require(not np.shares_memory(grad, param), "Return a new gradient array, independent of param.")
                for eps in (.1, .025):
                    grad = hw.numerical_grad(lambda: float(np.sum(param ** 3)), param, eps=eps)
                    suite.close(grad, 3 * original ** 2 + eps ** 2,
                                "Central differences of a cubic; use the supplied eps", atol=1e-8)
                    suite.require(np.array_equal(param, original), "Parameters were not restored.")
        suite.run(f"Vectors, matrices, tensors, central differences and restoration, seed {seed}", check)


def mlp_forward(suite, hw, cases):
    for index, raw in enumerate(cases["networks"]):
        def check(raw=raw):
            model = _network(hw, raw)
            probes = [raw["cache"]] + raw.get("probes", [])
            for expected in probes:
                x = np.array(expected["x"], float)
                suite.close(model.forward(x), expected["p"], "MLP.forward must compute from the current parameters")
                for key in ("x", "z1", "h1", "z2", "p"):
                    suite.require(hasattr(model, key), f"MLP.forward must store self.{key} for backward.")
                    suite.close(getattr(model, key), expected[key], f"Cached {key}")
            # Changing parameters must change even predictions at exact XOR corners.
            model.fc2.W[:] = 0
            for value in (0., -2., 2.):
                model.fc2.b[:] = value
                for expected in probes:
                    want = np.full(model.fc2.out_dim, 1 / (1 + np.exp(-value)))
                    suite.close(model.forward(np.array(expected["x"], float)), want,
                                "With zero output weights, p must equal sigmoid(bias); do not hard-code XOR")
            if model.fc2.out_dim == 1 and hasattr(hw, "predict"):
                X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
                model.fc2.b[:] = -2
                suite.close(hw.predict(model, X), np.zeros(4), "predict must use the model, not the XOR rule")
        suite.run(f"{raw['activation']} network, random parameters and parameter-sensitive predictions, case {index + 1}", check)


def mlp_backward(suite, hw, cases):
    for index, raw in enumerate(cases["networks"]):
        def check(raw=raw):
            model = _network(hw, raw)
            # Supply correct caches so a separate forward bug need not lose backward credit.
            for key, value in raw["cache"].items():
                setattr(model, key, np.array(value, float))
            upstream = np.array(raw["upstream"], float)
            original = upstream.copy()
            expected = raw["gradients"]
            suite.close(model.backward(upstream), expected["dx"], "MLP.backward dL/dx")
            for name in ("fc1", "fc2"):
                for parameter in ("W", "b"):
                    value = getattr(getattr(model, name), "d" + parameter)
                    suite.require(value is not None, f"Store self.{name}.d{parameter}.")
                    suite.close(value, expected[f"{name}.{parameter}"], f"{name}.d{parameter}")
            suite.require(np.array_equal(upstream, original), "Backward modified the upstream gradient.")
            for name, W, b in (("fc1", "W1", "b1"), ("fc2", "W2", "b2")):
                suite.close(getattr(model, name).W, raw[W], "Backward must not update weights", atol=0, rtol=0)
                suite.close(getattr(model, name).b, raw[b], "Backward must not update biases", atol=0, rtol=0)
        suite.run(f"{raw['activation']} chain rule with arbitrary upstream gradient, case {index + 1}", check)


def sgd(suite, hw, cases):
    for seed in cases["seeds"]:
        def check(seed=seed):
            rng = np.random.default_rng(seed)
            for lr in (0., .03, .5):
                layers = [SimpleNamespace(W=rng.normal(size=(o, i)), b=rng.normal(size=o),
                                          dW=rng.normal(size=(o, i)), db=rng.normal(size=o))
                          for i, o in ((3, 7), (7, 2), (2, 1))]
                before = [(layer.W.copy(), layer.b.copy(), layer.dW.copy(), layer.db.copy()) for layer in layers]
                model = SimpleNamespace(layers=lambda: layers)
                hw.sgd_step(model, lr)
                for layer, (W, b, dW, db) in zip(layers, before):
                    suite.close(layer.W, W - lr * dW, "SGD weight update")
                    suite.close(layer.b, b - lr * db, "SGD bias update")
                    suite.close(layer.dW, dW, "SGD must not change stored gradients", atol=0, rtol=0)
                    suite.close(layer.db, db, "SGD must not change stored gradients", atol=0, rtol=0)
        suite.run(f"All layers, weights and biases, including lr=0, seed {seed}", check)


def train_numpy(suite, hw, cases):
    for index, raw in enumerate(cases["training"]):
        def check(raw=raw):
            model = _network(hw, raw["network"])
            data = list(zip(np.array(raw["X"], float), np.array(raw["y"], float)))
            history = hw.train(model, data, hw.BCELoss(), lr=raw["lr"], epochs=raw["epochs"])
            suite.require(isinstance(history, list) and len(history) == raw["epochs"],
                          f"train must return a list of {raw['epochs']} mean epoch losses.")
            suite.close(history, raw["history"], "Mean loss before each sample update", atol=1e-8, rtol=1e-4)
            for layer, W, b in ((model.fc1, "W1", "b1"), (model.fc2, "W2", "b2")):
                suite.close(layer.W, raw["parameters"][W], "Weights after sequential SGD", atol=1e-8, rtol=1e-4)
                suite.close(layer.b, raw["parameters"][b], "Biases after sequential SGD", atol=1e-8, rtol=1e-4)
            if raw.get("xor"):
                suite.close([int(model.forward(x)[0] > .5) for x, _ in data], [0, 1, 1, 0],
                            "The seeded integration run should learn XOR")
        label = "Seeded XOR integration" if raw.get("xor") else "General data, multiple outputs and a short SGD trace"
        suite.run(f"{label}, case {index + 1}", check)


def run_task(key, hw, cases):
    suite = Suite()
    numpy_tests = {"1.1.loader": loader, "1.2.init": initialization,
                   "1.4.linear": linear, "1.4.bce": bce, "1.5.numgrad": numerical,
                   "1.6.forward": mlp_forward, "1.6.backward": mlp_backward,
                   "1.7.sgd": sgd, "1.7.train": train_numpy}
    if key.startswith("1.3."):
        suite.run("Activation checks", lambda: activation(suite, hw, cases, key.rsplit(".", 1)[1]))
    elif key in numpy_tests:
        suite.run("Assignment checks", lambda: numpy_tests[key](suite, hw, cases))
    else:
        import hw2_torch_checks as checks
        function = {"2.1.dataset": checks.dataset, "2.1.loaders": checks.loaders,
                    "2.2.model": checks.model, "2.3.bridge": checks.bridge,
                    "2.4.evaluate": checks.evaluate, "2.4.train": checks.train}[key]
        suite.run("Assignment checks", lambda: function(suite, hw, cases))
    # Setup is recorded only on failure; successful case details remain concise.
    suite.cases = [case for case in suite.cases if case["name"] not in
                   {"Assignment checks", "Activation checks"} or case["status"] != "PASSED"]
    return suite
