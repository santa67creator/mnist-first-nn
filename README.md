# MNIST: My First Neural Network

A fully connected network (784 → 500 → 250 → 10) built with TensorFlow for recognizing
handwritten MNIST digits. Based on Chapter 2 of a TensorFlow book and adapted
for modern TensorFlow using `tf.compat.v1`.

**Result:** ~98.3% test accuracy after 20 epochs (CPU).

<img width="1920" height="1080" alt="2026-10-05-124121_hyprshot" src="https://github.com/user-attachments/assets/c9a37870-aafc-47e9-a19f-7be04cffec50" />

<img width="1920" height="1080" alt="2026-10-05-124133_hyprshot" src="https://github.com/user-attachments/assets/4d39f246-4a09-4b93-9fe1-9fab9e9a598f" />


## Running the code
1. Download the MNIST dataset (4 `.gz` files) to `~/AI/datasets/mnist`
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python main.py`
