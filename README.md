# Number Identification Neural Network

A handwritten digit classifier (0–9) built from scratch using NumPy, trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). No ML frameworks — forward pass, backpropagation, and gradient descent are all implemented manually.

---

## Architecture

| Layer | Details |
|---|---|
| Input | 784 nodes (28×28 pixels, flattened) |
| Hidden | 64 nodes, ReLU activation |
| Output | 10 nodes, Softmax activation |

**Training:** Batch gradient descent with categorical cross-entropy loss.

---

## Setup

```bash
git clone https://github.com/DistancedTurtle/Number-Identification-NN.git
cd Number-Identification-NN
pip install -r requirements.txt
```

MNIST will be downloaded automatically on first run.

---

## Usage

**Train the model** (saves weights to `mnist_weights.npz`):
```bash
python main.py
```

**Evaluate saved weights** (skips training):
```bash
python main.py --test
```

---

## Results

Achieves ~**97.3% accuracy** on the MNIST test set after 2000 epochs.

---

## What I Learned

- Implementing forward and backward passes by hand (including the ReLU derivative mask and softmax + cross-entropy gradient)
- Why numerical stability matters in softmax (`exp` overflow without max-subtraction)
- Practical GitHub habits: structured commits, `.gitignore`, `requirements.txt`, README
