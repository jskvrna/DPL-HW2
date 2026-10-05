"""
DPL HW2 - Part 2: The same network in PyTorch  (3 points)
==========================================================

Same rules as in Part 1: fill in the TODO blocks, do not change names or
signatures, read the theory in README.md first.

    python test_hw2.py 2        # all tasks of Part 2
    python test_hw2.py 2.3      # a single task

In this part the data is NOISY XOR (many points scattered around the four
corners) and we train with MINI-BATCHES. A batch of inputs has the shape
(B, in_dim), a batch of targets (B, out_dim).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

CORNERS = torch.tensor([[0.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 0.0],
                        [1.0, 1.0]])
CORNER_LABELS = torch.tensor([0.0, 1.0, 1.0, 0.0])


# =============================================================================
#  Task 2.1 - Dataset and data loaders                     (2 x 0.25 = 0.5 pt)
# =============================================================================

class NoisyXOR(Dataset):
    """n points around the corners of the unit square, labelled by XOR.

    Every point: pick one of the 4 CORNERS uniformly at random, then add
    independent Gaussian noise N(0, noise^2) to both of its coordinates.
    The label is the XOR label of the chosen corner (CORNER_LABELS).

    Attributes:
        self.X: float32 tensor (n, 2)
        self.y: float32 tensor (n, 1)
    """

    def __init__(self, n=1000, noise=0.2, seed=0):
        ############################ TODO 2.1 (a) #############################
        # Create self.X and self.y as described above.
        # noise is the standard deviation; the variance is noise**2.
        # Scale torch.randn by noise, or pass noise as std to torch.normal.
        # Use ONLY  g = torch.Generator().manual_seed(seed)  as the source of
        # randomness (pass generator=g to torch.randint / torch.randn), so the
        # same seed always gives the same dataset.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 2.1 (a)")
        ############################## END TODO ###############################

    def __len__(self):
        ############################ TODO 2.1 (b) #############################
        # Return the number of samples.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 2.1 (b)")
        ############################## END TODO ###############################

    def __getitem__(self, i):
        ############################ TODO 2.1 (c) #############################
        # Return the pair (x, y) of sample i; shapes (2,) and (1,).
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 2.1 (c)")
        ############################## END TODO ###############################


def make_loaders(dataset, val_fraction=0.2, batch_size=32, seed=0):
    """Splits the dataset into train / validation parts and wraps them in loaders.

    Returns:
        train_loader: DataLoader over the training part, shuffled every epoch
        val_loader:   DataLoader over the validation part, NOT shuffled
    """
    ############################## TODO 2.1 (d) ###############################
    # 1. n_val = int(len(dataset) * val_fraction),  n_train = the rest
    # 2. split with torch.utils.data.random_split, passing
    #        generator=torch.Generator().manual_seed(seed)
    # 3. create the two DataLoaders with the given batch_size
    # -------------------------------------------------------------------------
    raise NotImplementedError("TODO 2.1 (d)")
    ################################ END TODO #################################


# =============================================================================
#  Task 2.2 - The model as an nn.Module                              (0.5 pt)
# =============================================================================

TORCH_ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}


class TorchMLP(nn.Module):
    """in_dim -> hidden_dim -> out_dim. Returns LOGITS (no sigmoid at the end)."""

    def __init__(self, in_dim=2, hidden_dim=16, out_dim=1, activation="relu"):
        super().__init__()
        ############################ TODO 2.2 (a) #############################
        # Create exactly these attributes:
        #     self.fc1 : nn.Linear(in_dim -> hidden_dim)
        #     self.act : the hidden activation, TORCH_ACTIVATIONS[activation]()
        #     self.fc2 : nn.Linear(hidden_dim -> out_dim)
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 2.2 (a)")
        ############################## END TODO ###############################

    def forward(self, x):
        """x: (B, in_dim)  ->  logits: (B, out_dim)"""
        ############################ TODO 2.2 (b) #############################
        # Return the logits. No sigmoid here - it is part of the loss
        # (nn.BCEWithLogitsLoss), see README.
        # ---------------------------------------------------------------------
        raise NotImplementedError("TODO 2.2 (b)")
        ############################## END TODO ###############################


# =============================================================================
#  Task 2.3 - Bridge: let autograd check your Part 1 gradients       (1.0 pt)
# =============================================================================

def torch_gradients(np_model, x, y, activation):
    """Computes the gradients of the Part 1 MLP with PyTorch autograd.

    Args:
        np_model: an MLP from hw2_numpy (has fc1.W, fc1.b, fc2.W, fc2.b)
        x:        numpy input,  shape (in_dim,)
        y:        numpy target, shape (out_dim,)
        activation: hidden activation name ("sigmoid", "tanh", or "relu");
                    must match the activation used by np_model

    Returns:
        dict with keys "fc1.W", "fc1.b", "fc2.W", "fc2.b"; the values are
        numpy float64 arrays with the same shapes as the numpy parameters.
        They must equal the gradients your MLP.backward computes for
        BCELoss on (x, y).
    """
    hidden_dim, in_dim = np_model.fc1.W.shape
    out_dim = np_model.fc2.W.shape[0]
    ################################ TODO 2.3 #################################
    # 1. Create a TorchMLP with the same sizes and activation=activation, then convert it
    #    to float64 with .double() (we want to compare to ~1e-10).
    # 2. Copy the numpy weights and biases into it (inside torch.no_grad(),
    #    e.g. model.fc1.weight.copy_(torch.from_numpy(...))).
    # 3. Make x, y float64 tensors with a batch dimension: shapes (1, in_dim)
    #    and (1, out_dim).
    # 4. loss = nn.BCEWithLogitsLoss()(model(x), y);  loss.backward()
    # 5. Read the .grad of the parameters and return them as numpy arrays.
    # -------------------------------------------------------------------------
    raise NotImplementedError("TODO 2.3")
    ################################ END TODO #################################


# =============================================================================
#  Task 2.4 - Evaluation and the training loop             (0.25 + 0.75 pt)
# =============================================================================

def evaluate(model, loader):
    """Accuracy (python float in [0, 1]) of the model on all samples of the loader.

    A sample is predicted as class 1 iff its logit is > 0 (i.e. probability > 0.5).
    """
    ############################## TODO 2.4 (a) ###############################
    # Switch the model to eval mode, disable gradient tracking
    # (torch.no_grad()), count the correct predictions over all batches.
    # -------------------------------------------------------------------------
    raise NotImplementedError("TODO 2.4 (a)")
    ################################ END TODO #################################


def train_torch(model, train_loader, val_loader, epochs=50, lr=0.5):
    """Trains the model with plain SGD and nn.BCEWithLogitsLoss.

    Returns:
        history: dict with two lists of length `epochs`
            "train_loss": mean training loss over the samples of each epoch
            "val_acc":    evaluate(model, val_loader) after each epoch
    """
    ############################## TODO 2.4 (b) ###############################
    # Create the optimizer torch.optim.SGD(model.parameters(), lr=lr) and the
    # loss nn.BCEWithLogitsLoss(). Then for every epoch:
    #   model.train()
    #   for every batch: zero_grad -> forward -> loss -> backward -> step
    #   record the mean train loss and the validation accuracy
    # The epoch loss is the mean over SAMPLES, not over batches: the last
    # batch can be smaller, so add loss.item() * len(xb) for every batch and
    # divide by the number of samples at the end of the epoch.
    # -------------------------------------------------------------------------
    raise NotImplementedError("TODO 2.4 (b)")
    ################################ END TODO #################################


def plot_decision_boundary(model, dataset, path="decision_boundary.png"):
    """(Provided) Saves a plot of the learned decision boundary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xx, yy = np.meshgrid(np.linspace(-0.75, 1.75, 300), np.linspace(-0.75, 1.75, 300))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(grid)).reshape(xx.shape).numpy()

    X, y = dataset.X.numpy(), dataset.y.numpy()[:, 0]
    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, prob, levels=20, cmap="RdBu_r", alpha=0.6)
    plt.contour(xx, yy, prob, levels=[0.5], colors="k")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="k", s=12)
    plt.title("Noisy XOR - decision boundary")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"saved {path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    dataset = NoisyXOR(n=1000, noise=0.2, seed=0)
    train_loader, val_loader = make_loaders(dataset, val_fraction=0.2, batch_size=32, seed=0)
    model = TorchMLP(in_dim=2, hidden_dim=16, out_dim=1, activation="relu")

    history = train_torch(model, train_loader, val_loader, epochs=50, lr=0.5)
    for epoch in range(0, len(history["val_acc"]), 10):
        print(f"epoch {epoch:3d}  train loss {history['train_loss'][epoch]:.4f}"
              f"  val acc {history['val_acc'][epoch]:.3f}")
    print(f"final val acc: {history['val_acc'][-1]:.3f}")

    # Compare autograd with your hand-written backprop from Part 1.
    import hw2_numpy
    np_model = hw2_numpy.MLP(2, 4, 1, "sigmoid", rng=np.random.default_rng(0))
    x, y = hw2_numpy.XOR_X[1], hw2_numpy.XOR_Y[1]
    np_model.backward(hw2_numpy.BCELoss().backward(np_model.forward(x), y))
    grads = torch_gradients(np_model, x, y, activation="sigmoid")
    print("\nPart 1 backprop vs. PyTorch autograd (relative error):")
    for name, layer in (("fc1", np_model.fc1), ("fc2", np_model.fc2)):
        print(f"  {name}.W: {hw2_numpy.relative_error(layer.dW, grads[name + '.W']):.2e}"
              f"   {name}.b: {hw2_numpy.relative_error(layer.db, grads[name + '.b']):.2e}")

    try:
        plot_decision_boundary(model, dataset)
    except ImportError:
        print("matplotlib not installed - skipping the plot")
