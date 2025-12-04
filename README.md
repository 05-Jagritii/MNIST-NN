# MNIST Neural Network — From Scratch (NumPy) + Streamlit Web App

This project implements a **fully connected neural network from scratch using only NumPy**, trained on the classic **MNIST handwritten digits dataset** (0–9).  
It also includes an interactive **Streamlit web app** where users can test the model in real time.

No TensorFlow layers.  
No PyTorch.  
No Keras models.  
Just pure **NumPy**, manual **forward pass**, **backpropagation**, **softmax**, and **SGD**.

---

## 🚀 Features

-  Neural Network implemented **100% from scratch**
-  Forward Propagation + Backpropagation + Gradient Descent
-  Softmax output for 10-class classification
-  Trained on 60,000 MNIST digit images
-  Streamlit UI to test predictions visually
-  Clean, readable, modular code suitable for portfolios

## Installation & Setup

### Clone the repository
  ```bash
  git clone https://github.com/05-Jagritii/MNIST-NN.git
  cd MNIST-NN
  ```
### Create a virtual environment
  Windows
  ```bash
  py -m venv .venv
.venv\Scripts\activate
  ```
  macOS/Linux
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
### Install all dependencies
  ```bash
  pip install -r requirements.txt
  ```

### Train the Neural Network
  ```bash
  py mnist_nn_from_scratch.py
  ```

### Run the Streamlit Web App
  ```bash
  streamlit run app.py
  ```

