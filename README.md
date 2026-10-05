# MNIST: My First Neural Network

A fully connected network (784 → 500 → 250 → 10) built with TensorFlow for recognizing
handwritten MNIST digits. Based on Chapter 2 of a TensorFlow book and adapted
for modern TensorFlow using `tf.compat.v1`.

**Result:** ~98.3% test accuracy after 20 epochs (CPU).

## Running the code
1. Download the MNIST dataset (4 `.gz` files) to `~/AI/datasets/mnist`
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python main.py`
