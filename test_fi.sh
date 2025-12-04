# 1. Train MNIST classifier
python3 examples/mnist_training_example.py

# 2. Build per-image Fisher dataset
mv mnist_model.pth checkpoints/mnist_trained_model.pt   # if needed
python3 examples/build_mnist_fisher_dataset.py

# 3. Train Fisher approximator
python3 examples/train_fisher_approximator_mnist.py

# 4. Run W&B calibration
python3 examples/calibrate_fisher_approximator_with_wan.py