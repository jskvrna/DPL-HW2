"""
DPL HW2 - Part 1: Backpropagation by hand in NumPy  (7 points)
================================================================

Fill in every block that looks like this:

    ################################ TODO x.y #################################
    # instructions ...
    # -------------------------------------------------------------------------
    raise NotImplementedError(...)        <-- replace this with your code
    ################################ END TODO #################################

Rules
-----
* Only NumPy is allowed in this file (no torch / jax / autograd / ...).
* Do not change names or signatures of the classes / functions.
* Do not change the code outside of the TODO blocks (you may add helpers).
* Read the theory for each task in README.md first.

Test your work locally:

    python test_hw2.py 1        # all tasks of Part 1
    python test_hw2.py 1.3      # a single task

Conventions (the batch size is always 1 in this part)
-----------------------------------------------------
    x       one input sample            shape (in_dim,)
    y       its target                  shape (out_dim,)
    W       weight matrix of a layer    shape (out_dim, in_dim)   (same as torch.nn.Linear)
    b       bias vector of a layer      shape (out_dim,)
    dL_dv   gradient dL/dv of the loss L w.r.t. a variable v.
            ALWAYS has the same shape as v.
"""

import numpy as np


# =============================================================================
#  Task 1.1 - Data and data loader                                   (0.5 pt)
# =============================================================================

# The whole XOR dataset: 4 inputs and their labels.
XOR_X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
XOR_Y = np.array([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])


class XORDataLoader:
    """Iterates over a dataset ONE sample at a time (batch size 1).

    Usage:
        loader = XORDataLoader(XOR_X, XOR_Y, shuffle=True, rng=np.random.default_rng(0))
        for epoch in range(3):
            for x, y in loader:          # one full loop = one epoch
                ...                      # x.shape == (2,), y.shape == (1,)

    Args:
        X:       inputs,  shape (N, in_dim)
        y:       targets, shape (N, out_dim)
        shuffle: if True, every epoch visits the samples in a new random order
        rng:     np.random.Generator used for shuffling
    """

    def __init__(self, X, y, shuffle=True, rng=None):
        assert len(X) == len(y), "X and y must have the same number of samples"
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.rng = rng if rng is not None else np.random.default_rng()

    def __len__(self):
        ############################ TODO 1.1 (a) #############################
        # Return the number of samples yielded in one epoch.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.1 (a)")
        ############################## END TODO ###############################

    def __iter__(self):
        ############################ TODO 1.1 (b) #############################
        # Yield all samples of ONE epoch as pairs (x, y), each sample exactly once.
        #   x = one row of self.X, shape (in_dim,)
        #   y = one row of self.y, shape (out_dim,)
        #
        # Order of the samples:
        #   shuffle=False -> 0, 1, ..., N-1
        #   shuffle=True  -> a random permutation; get it with exactly ONE call
        #                    self.rng.permutation(N) at the start of every epoch
        #                    (the tests compare your order with ours).
        #
        # This method is a *generator*: use `yield`, not `return`.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.1 (b)")
        ############################## END TODO ###############################


# =============================================================================
#  Task 1.2 - Linear layer: initialization                          (0.5 pt)
#  Task 1.4 - Linear layer: forward / backward                      (1.0 pt)
# =============================================================================

class Linear:
    """Fully connected layer:  z = W @ x + b.

    The layer does not remember anything between calls: `backward` gets the
    same input `x` that was given to `forward`.
    """

    def __init__(self, in_dim, out_dim, rng):
        self.in_dim = in_dim
        self.out_dim = out_dim

        ############################## TODO 1.2 ###############################
        # Initialize the parameters:
        #   self.W : Xavier (Glorot) normal initialization, shape (out_dim, in_dim).
        #            Draw it with exactly ONE call of
        #                rng.normal(0.0, std, size=(out_dim, in_dim))
        #            where std is the Xavier standard deviation (see README).
        #   self.b : zeros, shape (out_dim,)
        # Use only the given `rng` (never np.random.* directly).
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.2")
        ############################## END TODO ###############################

        # Gradients dL/dW and dL/db - filled in by MLP.backward, used by sgd_step.
        self.dW = None
        self.db = None

    def forward(self, x):
        """x: (in_dim,)  ->  z: (out_dim,)"""
        ############################ TODO 1.4 (a) #############################
        # Return z = W @ x + b.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.4 (a)")
        ############################## END TODO ###############################

    def backward(self, x, dL_dz):
        """Backward pass of the layer.

        Args:
            x:     the input that was given to forward,  shape (in_dim,)
            dL_dz: gradient of the loss w.r.t. the output z, shape (out_dim,)

        Returns:
            dL_dx: shape (in_dim,)
            dL_dW: shape (out_dim, in_dim)
            dL_db: shape (out_dim,)
        """
        ############################ TODO 1.4 (b) #############################
        # Apply the chain rule to z = W @ x + b (see README for the derivation).
        # Do not modify x or dL_dz in place.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.4 (b)")
        ############################## END TODO ###############################


# =============================================================================
#  Task 1.3 - Activation functions                         (3 x 0.5 = 1.5 pt)
#
#  forward(z)  returns  a = f(z)          element-wise, same shape as z
#  backward(z) returns  f'(z) = da/dz     element-wise, same shape as z
#
#  backward only returns the LOCAL derivative. Multiplying it with the
#  upstream gradient (the chain rule) is done in MLP.backward.
# =============================================================================

class Sigmoid:
    def forward(self, z):
        ############################ TODO 1.3 (a) #############################
        # Return sigmoid(z) = 1 / (1 + exp(-z)).
        # It must NOT overflow (no RuntimeWarning, no nan) for large |z|,
        # e.g. z = -1000 or z = 1000. See the hint in README.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.3 (a)")
        ############################## END TODO ###############################

    def backward(self, z):
        ############################ TODO 1.3 (b) #############################
        # Return the derivative of sigmoid at z.
        # Tip: it can be written using self.forward(z).
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.3 (b)")
        ############################## END TODO ###############################


class Tanh:
    def forward(self, z):
        ############################ TODO 1.3 (c) #############################
        # Return tanh(z)  (np.tanh is allowed).
        # It must NOT overflow (no RuntimeWarning, no nan) for large |z|,
        # e.g. z = -1000 or z = 1000 - writing it with np.exp does.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.3 (c)")
        ############################## END TODO ###############################

    def backward(self, z):
        ############################ TODO 1.3 (d) #############################
        # Return the derivative of tanh at z.
        # It must NOT overflow for large |z| either: write it using tanh
        # (forms with np.cosh or np.exp overflow at z = 1000).
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.3 (d)")
        ############################## END TODO ###############################


class ReLU:
    def forward(self, z):
        ############################ TODO 1.3 (e) #############################
        # Return max(0, z) element-wise.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.3 (e)")
        ############################## END TODO ###############################

    def backward(self, z):
        ############################ TODO 1.3 (f) #############################
        # Return the derivative of ReLU at z as floats (1.0 or 0.0).
        # Convention: the derivative at exactly z = 0 is 0.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.3 (f)")
        ############################## END TODO ###############################


ACTIVATIONS = {"sigmoid": Sigmoid, "tanh": Tanh, "relu": ReLU}


# =============================================================================
#  Task 1.4 - Binary cross-entropy loss                              (0.5 pt)
# =============================================================================

class BCELoss:
    """Binary cross-entropy between predicted probabilities p and targets y:

        L = -mean( y * log(p) + (1 - y) * log(1 - p) )

    The mean is over all elements (here out_dim = 1, so it is a single term).
    To avoid log(0), p is first clipped to [EPS, 1 - EPS] - in BOTH forward
    and backward.
    """

    EPS = 1e-7

    def forward(self, p, y):
        """p, y: (out_dim,)  ->  L: python float"""
        ############################ TODO 1.4 (c) #############################
        # Return the loss as a python float (use float(...)).
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.4 (c)")
        ############################## END TODO ###############################

    def backward(self, p, y):
        """p, y: (out_dim,)  ->  dL_dp: (out_dim,)"""
        ############################ TODO 1.4 (d) #############################
        # Return dL/dp, using the clipped p. Do not forget the 1/n of the mean.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.4 (d)")
        ############################## END TODO ###############################


# =============================================================================
#  Task 1.5 - Numerical gradient (gradient check)                     (0.5 pt)
# =============================================================================

def numerical_grad(f, param, eps=1e-5):
    """Estimates df/dparam with central differences.

    Args:
        f:     function with NO arguments returning a float. It reads `param`
               (e.g. f = lambda: loss_fn.forward(model.forward(x), y)).
        param: numpy array that f depends on. You may change its entries
               temporarily IN PLACE, but every entry must hold exactly its
               original value when the function returns.
        eps:   step size (epsilon in the README formula).

    Returns:
        grad: new array of the same shape as param, where
              grad[i] = (f(param[i] + eps) - f(param[i] - eps)) / (2 * eps)
              with ONLY entry i changed (all other entries unchanged).
    """
    ################################ TODO 1.5 #################################
    # Loop over all entries of param (np.ndindex(param.shape) gives all
    # index tuples, also for 2-D arrays), perturb one entry at a time,
    # call f() twice and restore the entry.
    # Restore it by saving the old value first and assigning it back:
    # adding and subtracting eps does NOT give back exactly the same float.
    # -------------------------------------------------------------------------
    raise NotImplementedError("TODO 1.5")
    ################################ END TODO #################################


def relative_error(a, b):
    """max |a - b| / (|a| + |b|), element-wise, safe for zeros."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def check_gradients(model, x, y, loss_fn=None):
    """(Provided) Compares MLP.backward with numerical_grad on one sample.

    Returns a dict {"fc1.W": rel_err, ...}. A correct implementation gives
    errors below 1e-6 (usually 1e-7 or smaller); anything above 1e-4 is almost
    surely a bug.
    """
    loss_fn = loss_fn if loss_fn is not None else BCELoss()
    p = model.forward(x)
    model.backward(loss_fn.backward(p, y))
    analytic = {f"{name}.{par}": getattr(layer, "d" + par).copy()
                for name, layer in (("fc1", model.fc1), ("fc2", model.fc2))
                for par in ("W", "b")}

    def f():
        return loss_fn.forward(model.forward(x), y)

    errors = {}
    for name, layer in (("fc1", model.fc1), ("fc2", model.fc2)):
        for par in ("W", "b"):
            numeric = numerical_grad(f, getattr(layer, par))
            errors[f"{name}.{par}"] = relative_error(analytic[f"{name}.{par}"], numeric)
    return errors


# =============================================================================
#  Task 1.6 - The MLP: forward and backward, wired by hand           (1.5 pt)
#
#      x --fc1--> z1 --act1--> h1 --fc2--> z2 --act2(sigmoid)--> p --BCE--> L
# =============================================================================

class MLP:
    """Two-layer perceptron  in_dim -> hidden_dim -> out_dim, sigmoid output."""

    def __init__(self, in_dim=2, hidden_dim=4, out_dim=1, activation="sigmoid", rng=None):
        # (Provided - do not change.)
        rng = rng if rng is not None else np.random.default_rng()
        self.activation = activation
        self.fc1 = Linear(in_dim, hidden_dim, rng)
        self.act1 = ACTIVATIONS[activation]()
        self.fc2 = Linear(hidden_dim, out_dim, rng)
        self.act2 = Sigmoid()

    def layers(self):
        """(Provided) Layers with parameters."""
        return [self.fc1, self.fc2]

    def forward(self, x):
        """x: (in_dim,)  ->  p: (out_dim,), probabilities in (0, 1)."""
        ############################ TODO 1.6 (a) #############################
        # Compute the forward pass step by step using the modules
        # self.fc1, self.act1, self.fc2, self.act2 and STORE the intermediate
        # results on self - backward needs them:
        #     self.x, self.z1, self.h1, self.z2, self.p
        # Return self.p.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.6 (a)")
        ############################## END TODO ###############################

    def backward(self, dL_dp):
        """Backpropagates dL/dp through the network.

        Uses the values stored by the last call of forward.
        Stores the parameter gradients in
            self.fc1.dW, self.fc1.db, self.fc2.dW, self.fc2.db
        and returns dL/dx, shape (in_dim,).
        """
        ############################ TODO 1.6 (b) #############################
        # Go through the graph from the end to the start:
        #     dL_dp -> dL_dz2 -> dL_dh1 -> dL_dz1 -> dL_dx
        # * activation:   dL_dz = dL_da * act.backward(z)      (chain rule)
        # * linear layer: dL_din, dL_dW, dL_db = fc.backward(input_of_fc, dL_dout)
        # Store the gradients of both layers and return dL_dx.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 1.6 (b)")
        ############################## END TODO ###############################


# =============================================================================
#  Task 1.7 - SGD and the training loop                              (1.0 pt)
# =============================================================================

def sgd_step(model, lr):
    """One step of gradient descent on all layers of the model."""
    ############################## TODO 1.7 (a) ###############################
    # For every layer in model.layers():  W <- W - lr * dW,  b <- b - lr * db
    # -------------------------------------------------------------------------
    raise NotImplementedError("TODO 1.7 (a)")
    ################################ END TODO #################################


def train(model, loader, loss_fn, lr, epochs):
    """Trains the model with SGD, one sample at a time.

    Returns:
        history: list of length `epochs`; history[e] is the MEAN loss over all
                 samples of epoch e (each loss computed before that sample's
                 update step).
    """
    ############################## TODO 1.7 (b) ###############################
    # For every epoch, for every (x, y) from the loader:
    #   1. forward pass  -> p
    #   2. loss          -> L        (remember it for the epoch mean)
    #   3. dL/dp         -> loss_fn.backward
    #   4. backward pass -> model.backward
    #   5. update        -> sgd_step
    # -------------------------------------------------------------------------
    raise NotImplementedError("TODO 1.7 (b)")
    ################################ END TODO #################################


def predict(model, X):
    """(Provided) Class predictions 0/1 for all rows of X."""
    return np.array([model.forward(x)[0] > 0.5 for x in X], dtype=int)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    model = MLP(in_dim=2, hidden_dim=4, out_dim=1, activation="sigmoid", rng=rng)
    loader = XORDataLoader(XOR_X, XOR_Y, shuffle=True, rng=rng)
    loss_fn = BCELoss()

    print("Gradient check (should be < 1e-6):")
    for name, err in check_gradients(model, XOR_X[1], XOR_Y[1], loss_fn).items():
        print(f"  {name}: {err:.2e}")

    history = train(model, loader, loss_fn, lr=0.5, epochs=1000)
    for epoch in range(0, len(history), 100):
        print(f"epoch {epoch:4d}  loss {history[epoch]:.4f}")
    print(f"epoch {len(history) - 1:4d}  loss {history[-1]:.4f}")

    print("\n  x1  x2 | target  p")
    for x, y in zip(XOR_X, XOR_Y):
        print(f"  {x[0]:.0f}   {x[1]:.0f}  |   {y[0]:.0f}    {model.forward(x)[0]:.3f}")
    print("accuracy:", np.mean(predict(model, XOR_X) == XOR_Y[:, 0]))
