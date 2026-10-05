# DPL HW2: Backpropagation by Hand, then with Autograd

In this homework you train a small neural network to solve **XOR**, the classic problem that a single linear layer cannot solve. You do it twice:

1. **Part 1, NumPy (7 pt).** You write every piece yourself: the data loader, initialization, layers, loss, forward pass, **backward pass**, a gradient check and the training loop. There is no magic: you see exactly which gradient flows where.
2. **Part 2, PyTorch (3 pt).** You build the same thing using `Dataset`, `DataLoader`, `nn.Module`, autograd and an optimizer. You also let autograd verify your Part 1 gradients.

| Part | Task | What you implement | Points |
|------|------|--------------------|-------:|
| 1 | 1.1 | `XORDataLoader` | 0.5 |
| 1 | 1.2 | `Linear.__init__`: Xavier initialization | 0.5 |
| 1 | 1.3 | `Sigmoid`, `Tanh`, `ReLU`: forward and derivative | 3 × 0.5 |
| 1 | 1.4 | `Linear` forward/backward (1.0), `BCELoss` forward/backward (0.5) | 1.5 |
| 1 | 1.5 | `numerical_grad`: gradient check | 0.5 |
| 1 | 1.6 | `MLP.forward` (0.5), `MLP.backward` (1.0) | 1.5 |
| 1 | 1.7 | `sgd_step` (0.25), `train` (0.75) | 1.0 |
| 2 | 2.1 | `NoisyXOR` dataset (0.25), `make_loaders` (0.25) | 0.5 |
| 2 | 2.2 | `TorchMLP` (`nn.Module`) | 0.5 |
| 2 | 2.3 | `torch_gradients`: autograd vs. your backprop | 1.0 |
| 2 | 2.4 | `evaluate` (0.25), `train_torch` (0.75) | 1.0 |
| | | **Total** | **10** |

---

## Requirements

- **Python ≥ 3.10**
- **NumPy ≥ 1.24**, needed for Part 1
- **PyTorch ≥ 2.0**, needed for Part 2. A CPU build is enough; no GPU is needed.
- **matplotlib**, only for the decision-boundary plot in Part 2. It is in `requirements.txt`; without it, `hw2_torch.py` just skips the plot.

```bash
pip install -r requirements.txt
```

You can also use a virtual environment or conda if you prefer. On Linux, a smaller CPU-only PyTorch can be installed with `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Files

| File | Purpose |
|------|---------|
| `hw2_numpy.py` | **Part 1**: your NumPy implementation (TODO blocks) |
| `hw2_torch.py` | **Part 2**: your PyTorch implementation (TODO blocks) |
| `test_hw2.py` | local tests for every task, using the same checks as BRUTE |
| `make_submission.py` | creates `hw2_submission.zip` for upload |
| `requirements.txt` | Python packages |
| `tests/` | test helpers and the public test inputs with independently computed expected values (`hw2_cases.json`); you don't need to edit or read them |

## How to work

Every place where you write code is a clearly marked block:

```python
        ############################ TODO 1.3 (a) #############################
        # Return sigmoid(z) = 1 / (1 + exp(-z)).
        # ...
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.3 (a)")      # <- replace with your code
        ############################## END TODO ###############################
```

Go through the tasks in order. **After each task, run its tests**:

```bash
python test_hw2.py 1.1        # one task
python test_hw2.py 1          # all of Part 1
python test_hw2.py            # everything
```

```
[1.1] XORDataLoader ................... PASSED   0.5 / 0.5 pt
[1.2] Linear initialization ........... FAILED   0 / 0.5 pt
   Xavier normal weights; use exactly one call to the supplied RNG: values differ (maximum absolute error 0.0478084; atol=1e-12, rtol=1e-12).
   Expected excerpt: [ 0.079519, -0.08355 ,  0.405039,  0.066345, -0.338787,  0.228693]
   Actual excerpt:   [ 0.088905, -0.093412,  0.452847,  0.074176, -0.378775,  0.255686]
   3 checks failed; save the JSON report for individual case details.
[1.3] Sigmoid ......................... NOT IMPLEMENTED   0 / 0.5 pt
   Implementation needed: TODO 1.3 (a).
```

Tests of later tasks need some earlier ones. For example, `MLP.forward` uses your layers. If a test says `NOT IMPLEMENTED`, the line below it (`Implementation needed: TODO x.y`) tells you which TODO it is waiting for.

When Part 1 is done, run the whole pipeline. It runs a gradient check and then trains on XOR:

```bash
python hw2_numpy.py
python hw2_torch.py       # Part 2: trains, compares gradients, saves decision_boundary.png
```

## Rules

- Don't change the names or signatures of the provided classes and functions, and don't change code outside the TODO blocks. You may add your own helper functions.
- **Part 1 is NumPy only.** Importing `torch`, `jax`, `autograd`, `tensorflow`, `keras`, `cupy` and similar libraries in `hw2_numpy.py` is forbidden throughout the file, including inside functions and conditional branches. Tests inspect the source and block dynamic imports during execution. A forbidden import gives **zero points for all of Part 1**, even if its exception is caught. Part 2 is graded independently.
- Where a TODO prescribes how to use the random generator (`rng`, `torch.Generator`), follow it exactly. The tests compare your random numbers with ours.
- Don't hard-code anything. See below.

## Submission and grading

Create the archive and upload **`hw2_submission.zip`** to BRUTE:

```bash
python make_submission.py
```

The script zips `hw2_numpy.py` and `hw2_torch.py`, runs the local tests, and warns you about syntax errors, TODOs that are not implemented yet, and forbidden imports in Part 1. Use `--skip-tests` to only create the zip.

BRUTE runs **the same check code as the local tests**, with additional seeded inputs and expected values computed from an independent reference. Local tests already cover the same dimensions, edge cases, and training rules. Passing the complete local suite is therefore a strong indication that your upload will pass; additional seeds can still expose code that only works for particular values. For example:

- Layers and the MLP are tested with other sizes (such as `Linear(7, 3)` or `MLP(3, 5, 2, "tanh")`), random weights and random inputs, including XOR-shaped networks with random weights.
- The activations are tested on arrays of any shape, including the edge cases mentioned in the TODOs (large `|z|`, `z = 0`).
- `numerical_grad` is tested on functions other than your network.
- Training is checked against reference losses and parameter updates, with multiple activations and arbitrary labels. PyTorch tests include a smaller final batch: the epoch loss is the mean over **samples**, so each batch loss must be weighted by its actual sample count.

So if your code is correct *in general*, the hidden tests pass. A solution that only matches the numbers in `test_hw2.py`, or that special-cases the four XOR inputs, will fail. Points per task are all or nothing, as listed in the table above.

Each scoring unit runs in a separate process with a time limit, so one unfinished function or infinite loop does not stop all grading. The upload report shows the total score, points for each unit, and feedback for failed checks. To save all local feedback, run `python test_hw2.py --json results.json`.

For individual diagnostics in the terminal, use `python test_hw2.py 2.4 --verbose`. Training checks report gradient clearing, training/evaluation modes, and sample-weighted loss averaging separately, so one mistake does not hide another.

---

# Part 1: NumPy (7 points)

### Notation and shapes

In Part 1 the batch size is always **1**: the network sees one sample at a time, and every sample is a 1‑D vector.

| Symbol | Meaning | Shape |
|--------|---------|-------|
| $x$ | input sample | `(in_dim,)` |
| $y$ | target | `(out_dim,)` |
| $W$ | weights of a layer | `(out_dim, in_dim)` (same as `torch.nn.Linear`) |
| $b$ | bias of a layer | `(out_dim,)` |
| `dL_dv` | $\partial L / \partial v$, the gradient of the loss w.r.t. $v$ | **always the same shape as $v$** |

The network is fixed:

```
x ──fc1──► z1 ──act1──► h1 ──fc2──► z2 ──sigmoid──► p ──BCE(p, y)──► L
(2)        (4)          (4)         (1)              (1)
```

$$z_1 = W_1 x + b_1,\quad h_1 = f(z_1),\quad z_2 = W_2 h_1 + b_2,\quad p = \sigma(z_2),\quad L = \mathrm{BCE}(p, y)$$

The hidden activation $f$ is sigmoid by default (tanh and ReLU also work). The output activation is always a sigmoid, so $p \in (0, 1)$ is the probability of class 1.

**Why a hidden layer?** XOR is not linearly separable: no single line splits $\{(0,1), (1,0)\}$ from $\{(0,0), (1,1)\}$. The hidden layer maps the four points into a space where a line *can* separate them.

---

## Task 1.1: Data loader (0.5 pt)

The XOR dataset has 4 samples (`XOR_X`, `XOR_Y` are provided). A **data loader** decides in what order the model sees the samples:

- An **epoch** is one pass over the whole dataset. Every sample appears **exactly once**.
- With **shuffling**, the order is a new random permutation in every epoch. Without it, SGD sees the same sequence every time, which can create cycles and slow training down. With shuffling, every epoch sees a different sequence.
- A loader is an *iterable*: `for x, y in loader:` runs one epoch. In Python the easiest way to write `__iter__` is as a **generator** function, which uses `yield` to hand out items one at a time:

  ```python
  def __iter__(self):
      for item in something:
          yield item          # the for-loop of the caller receives `item`
  ```

**Implement** `__len__` and `__iter__`. Reproducibility matters: with `shuffle=True`, every epoch must call `self.rng.permutation(N)` exactly once, so the same seed always gives the same order. With `shuffle=False`, the order is `0, 1, ..., N-1` and the `rng` is not used at all.

---

## Task 1.2: Initialization (0.5 pt)

**Why not zeros?** If all weights start equal, all hidden neurons compute the same function and receive the same gradient, so they stay identical forever. This is called *symmetry*. Random initialization breaks the symmetry.

**How large should the weights be?** If the weights are too large, the pre-activations $z$ become large, sigmoid/tanh saturate, and the gradients vanish. If they are too small, the signal fades from layer to layer. **Xavier (Glorot) initialization** keeps the variance of the activations, and of the gradients, roughly constant across layers:

$$W_{ij} \sim \mathcal{N}\!\left(0,\ \sigma^2\right), \qquad \sigma = \sqrt{\frac{2}{n_\text{in} + n_\text{out}}}, \qquad b = 0.$$

The biases can start at zero, because the random $W$ already breaks the symmetry.

**Implement** `Linear.__init__`, drawing the weights with a single `rng.normal(0.0, std, size=(out_dim, in_dim))` call.

---

## Task 1.3: Activation functions (1.5 pt)

Each activation module has two methods:

- `forward(z)` returns $f(z)$, element-wise.
- `backward(z)` returns the **local derivative** $f'(z)$, element-wise, for the same `z`.

The module does **not** apply the chain rule. You do that yourself in `MLP.backward`, by multiplying the incoming gradient by $f'(z)$. This keeps every step of backpropagation visible.

| | $f(z)$ | used where |
|--|--|--|
| Sigmoid | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | output layer (probability), classic hidden layer |
| Tanh | $\tanh(z) = \dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$ | hidden layer, zero-centered |
| ReLU | $\max(0, z)$ | the default hidden layer in modern networks |

Derive the derivatives yourself. Each one is a one-liner. All activations are also tested at large $|z|$ (e.g. $z = \pm 1000$) and must not overflow there: use `np.tanh` for tanh and its derivative, not `np.exp` or `np.cosh`.

<details><summary>Hint: derivatives</summary>

- $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$
- $\tanh'(z) = 1 - \tanh^2(z)$
- $\mathrm{ReLU}'(z) = 1$ for $z > 0$, else $0$. At $z = 0$ the derivative does not exist, so we use the convention $0$.
</details>

**Numerically stable sigmoid.** `np.exp(-z)` overflows for `z = -1000` (it gives `inf` and a `RuntimeWarning`). Note that
$\sigma(z) = \frac{1}{1 + e^{-z}}$ for $z \ge 0$ and $\sigma(z) = \frac{e^{z}}{1 + e^{z}}$ for $z < 0$. Both forms are equal, but each one only calls `exp` of a non-positive number. Careful: `np.where(cond, A, B)` evaluates **both** `A` and `B` on every element. Compute `e = np.exp(-np.abs(z))` first, then combine the two forms with `np.where`.

---

## Task 1.4: Linear layer and BCE loss (1.5 pt)

### Linear layer: $z = W x + b$

Forward is just the formula. For backward, assume we already know the upstream gradient $\frac{\partial L}{\partial z}$ (shape `(out_dim,)`) and want the gradients w.r.t. the layer's input and parameters. Written element-wise, $z_i = \sum_j W_{ij} x_j + b_i$, so by the chain rule:

$$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial z_i}\, x_j, \qquad
\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial z_i}, \qquad
\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial z_i}\, W_{ij}.$$

Now write these **without loops**, using NumPy matrix operations. The shapes tell you a lot: `dL_dW` must be `(out_dim, in_dim)` and `dL_dx` must be `(in_dim,)`.

<details><summary>Hint: vectorized</summary>

`dL_dW = np.outer(dL_dz, x)`, `dL_db = dL_dz`, `dL_dx = W.T @ dL_dz`
</details>

`Linear.backward(x, dL_dz)` receives the **same input `x`** that was given to forward. The layer stores nothing; the MLP keeps the intermediate values.

### Binary cross-entropy

For a probability $p$ and a target $y \in \{0, 1\}$:

$$L(p, y) = -\big[\,y \log p + (1 - y) \log(1 - p)\,\big]$$

This is the negative log-likelihood of a Bernoulli distribution. It is small when $p$ is close to $y$ and grows to infinity as $p$ approaches the wrong extreme. For vectors, take the **mean** over the elements. Before using $p$, clip it to $[\varepsilon, 1 - \varepsilon]$ with $\varepsilon = 10^{-7}$ (`BCELoss.EPS`), in both forward and backward, so that `log(0)` and division by 0 never happen.

<details><summary>Hint: derivative</summary>

$\dfrac{\partial L}{\partial p} = -\dfrac{y}{p} + \dfrac{1 - y}{1 - p}$, and with the mean over $n$ elements, divide by $n$.
Interesting fact: combined with the sigmoid, $\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial p}\,\sigma'(z_2) = p - y$. Check that your backward gives this.
</details>

---

## Task 1.5: Numerical gradient, the gradient check (0.5 pt)

How do you know your backward is correct? Compare it with a **numerical estimate**. For each parameter entry $\theta_i$:

$$\frac{\partial f}{\partial \theta_i} \approx \frac{f(\theta + \varepsilon\,e_i) - f(\theta - \varepsilon\,e_i)}{2\varepsilon}$$

- $\varepsilon$ is the **step size**, a single small number (the `eps` argument in the code).
- $e_i$ is the **one-hot vector** for entry $i$: a 1 at position $i$ and 0 everywhere else. It selects *which* entry is nudged.

Together, $\theta + \varepsilon\,e_i$ means "$\theta$ with **only entry $i$** increased by $\varepsilon$; all other entries stay the same":

```
θ             = [ 0.5,  -0.3,      0.8 ]
e_1           = [ 0,     1,        0   ]      (one-hot, i = 1)
θ + ε·e_1     = [ 0.5,  -0.3 + ε,  0.8 ]
```

Why not just $\theta + \varepsilon$? That would shift **all** entries at once, and you would measure how $f$ changes when everything moves together (the sum of all partial derivatives), not the single entry $\partial f / \partial \theta_i$ you need. So you repeat the estimate for every entry, one at a time. For a matrix such as `W` it works the same way; $e_i$ is then a matrix with a single 1.

This is the **central difference**. Its error is $O(\varepsilon^2)$, compared with $O(\varepsilon)$ for the one-sided $\frac{f(\theta + \varepsilon\,e_i) - f(\theta)}{\varepsilon}$. In practice $\varepsilon \approx 10^{-5}$ with float64 gives about 7–10 correct digits.

`numerical_grad(f, param, eps)` gets a function `f()` with no arguments that reads `param`, for example `lambda: loss_fn.forward(model.forward(x), y)`. You change one entry of `param` **in place**, call `f()`, and restore the entry. Save the old value and assign it back; adding and subtracting `eps` does not always give back exactly the same float. It is slow (2 forward passes per parameter), so it is only used for checking, never for training.

The provided `check_gradients(model, x, y)` uses your `numerical_grad` to compare all four parameter gradients of the MLP with your backward pass. **Use it while working on Task 1.6.** Relative errors below `1e-6` (usually `1e-7` or smaller) mean the gradient is correct; anything above `1e-4` is almost surely a bug.

---

## Task 1.6: The MLP, forward and backward wired by hand (1.5 pt)

**Forward:** call the modules one after another and **store every intermediate result on `self`** (`self.x, self.z1, self.h1, self.z2, self.p`). The backward pass needs them.

**Backward = the chain rule, applied from the end to the start.** Starting from $\frac{\partial L}{\partial p}$ (given by the loss), every step turns the gradient w.r.t. a module's output into the gradient w.r.t. its input:

```
dL_dp ──act2──► dL_dz2 ──fc2──► dL_dh1 ──act1──► dL_dz1 ──fc1──► dL_dx
                         │                                 │
                         └─► fc2.dW, fc2.db                └─► fc1.dW, fc1.db
```

- through an **activation** $a = f(z)$: $\dfrac{\partial L}{\partial z} = \dfrac{\partial L}{\partial a} \odot f'(z)$, an element-wise product with your `backward(z)`
- through a **linear layer**: `fc.backward(input_of_fc, dL_dout)` gives the gradient for its input and its parameters

Store the parameter gradients in `fc1.dW, fc1.db, fc2.dW, fc2.db` and return `dL_dx`.

This is exactly what `loss.backward()` does in PyTorch. It records the graph during forward and walks it in reverse. Here you write the walk yourself.

---

## Task 1.7: SGD and the training loop (1.0 pt)

**Stochastic gradient descent** with batch size 1 updates the parameters after **every sample**:

$$\theta \leftarrow \theta - \eta\, \frac{\partial L}{\partial \theta}$$

where $\eta$ is the learning rate. One training step is:

1. forward: $p = \text{model}(x)$
2. loss: $L(p, y)$ (store it, to report the epoch mean)
3. $\partial L / \partial p$ from the loss
4. backward through the model
5. SGD update

`train` returns the **mean loss of every epoch**. With the default settings (2 → 4 → 1, sigmoid, lr = 0.5, 1000 epochs) the network solves XOR. After the first few hundred epochs the loss usually drops sharply, once the hidden units "find" a useful representation.

---

# Part 2: PyTorch (3 points)

Now the data is **noisy XOR**: many points scattered around the four corners. Training uses **mini-batches**, with tensors of shape `(B, 2)` instead of `(2,)`. Watch how much code disappears.

## Task 2.1: `Dataset` and `DataLoader` (0.5 pt)

PyTorch splits the work of loading data in two:

- A **`Dataset`** says *what* the data is. It needs `__len__` and `__getitem__(i) -> (x_i, y_i)`.
- A **`DataLoader`** says *how* the data is served. It batches the samples (stacking them into `(B, ...)` tensors), shuffles them every epoch (`shuffle=True`), and can load in parallel. It does what your `XORDataLoader` did, plus batching.

Generating noisy XOR: for each of the $n$ points, choose a corner $c \in \{(0,0), (0,1), (1,0), (1,1)\}$ uniformly at random. The point is $x = c + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \text{noise}^2 I)$, and its label is the XOR of $c$. Here **`noise` is the standard deviation**, and `noise**2` is the variance. Use a `torch.Generator` seeded with `seed` for all randomness. Don't touch the global RNG; that's what makes the dataset reproducible.

We **split** the data into a training and a validation part (`random_split`). The model learns on the training part, and the validation accuracy tells us how well it generalizes to points it has never seen. Only the training loader is shuffled. Shuffling the validation set has no effect on the result, so it isn't needed.

## Task 2.2: The model as an `nn.Module` (0.5 pt)

An `nn.Module` registers every layer assigned as an attribute in `__init__`, so `model.parameters()` finds all their weights automatically. You only write `forward`, and autograd derives backward for you.

The model returns **logits** $z_2$, not probabilities. The sigmoid is part of the loss, `nn.BCEWithLogitsLoss`, which computes $\mathrm{BCE}(\sigma(z), y)$ in one numerically stable formula:

$$L = \max(z, 0) - z\,y + \log\!\big(1 + e^{-|z|}\big)$$

This never overflows and needs no clipping, which is why PyTorch models for binary classification usually end with a plain `Linear` layer. To get probabilities, apply `torch.sigmoid(logits)`. To get classes, check `logits > 0`, which is equivalent to $p > 0.5$.

## Task 2.3: Autograd checks your Part 1 (1.0 pt)

**Autograd** records every operation on tensors with `requires_grad=True` into a graph during the forward pass. `loss.backward()` then walks that graph in reverse and fills `param.grad`. That's exactly the chain rule you coded in Task 1.6, done automatically for any graph.

`torch_gradients(np_model, x, y, activation)` copies the NumPy weights of your Part 1 MLP into a `TorchMLP` and lets autograd compute the gradients for the same sample. Pass the hidden activation name (`"sigmoid"`, `"tanh"`, or `"relu"`) explicitly; it must match the NumPy model's activation. Use this argument when constructing the `TorchMLP`. The gradients must match what your `MLP.backward` computes, up to about `1e-10`. Use float64 (`model.double()`), because float32 only has about 7 significant digits. `python hw2_torch.py` prints the comparison.

Note that the NumPy model is `sigmoid → BCE` while the torch model is `logits → BCEWithLogitsLoss`. Mathematically it is the same function, so the gradients are the same.

## Task 2.4: Evaluation and the training loop (1.0 pt)

The standard PyTorch training loop:

```python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()          # 1. clear the old gradients (they ACCUMULATE otherwise)
        loss = loss_fn(model(xb), yb)  # 2. forward + loss
        loss.backward()                # 3. autograd: fills p.grad for every parameter
        optimizer.step()               # 4. update: p -= lr * p.grad for every parameter (SGD)
```

- **`zero_grad`** is required because `.backward()` *adds* to `.grad`. Without it, the gradients of all previous steps pile up.
- **`torch.optim.SGD`** performs exactly the update you wrote in `sgd_step` (Task 1.7), now averaged over a mini-batch of 32 samples instead of one. Other optimizers (momentum, Adam, ...) only change the `step()` rule; the loop stays the same.
- **The epoch loss is the mean over samples.** The last batch can be smaller than 32, so accumulate `loss.item() * len(xb)` and divide by the number of samples, instead of averaging the batch losses.
- **`model.train()` / `model.eval()`** switch layers such as dropout and batch norm between training and inference behaviour. Our model has none, but it's a habit worth building.
- **`torch.no_grad()`** during evaluation: no graph is recorded, which is faster and uses less memory.

With the defaults (2 → 16 → 1, ReLU, SGD with lr = 0.5, batch size 32, 50 epochs) you should get **≥ 95 % validation accuracy**. Because of the noise, a few points end up on the "wrong" side of the decision boundaries, so 100 % is usually not reached (with some seeds the validation split happens to contain none of them). `python hw2_torch.py` saves a plot of the learned decision boundary to `decision_boundary.png`.

---

## Tips

- Shape errors are the most common bug. Print `.shape` freely. Every `dL_dv` has the shape of `v`.
- If `check_gradients` reports a large error for only one parameter, the bug is in the step that produces that gradient.
- `np.outer`, `@` and `.T` are all you need for the linear layer.
- Training "works but slowly", or gets stuck at loss ≈ 0.69 (= $\ln 2$, a constant prediction of 0.5): check the sign in `sgd_step` and the derivatives in backward.
