import numpy as np
import streamlit as st

from mnist_nn_from_scratch import (
    NeuralNetwork,
    load_mnist,
    one_hot,
)


@st.cache_data
def get_data():
    X_train, y_train, X_test, y_test = load_mnist()
    y_train_oh = one_hot(y_train, 10)
    return X_train, y_train, y_train_oh, X_test, y_test


@st.cache_resource
def train_model():
    X_train, y_train, y_train_oh, X_test, y_test = get_data()

    model = NeuralNetwork(
        input_dim=784,
        hidden_layers=[128, 64],
        output_dim=10,
        activation="relu",
        lr=0.01,
    )

    model.fit(X_train, y_train_oh, epochs=8, batch_size=64)

    return model, X_test, y_test


def main():
    st.set_page_config(page_title="MNIST NN from Scratch", page_icon="🧠")
    st.title("🧠 MNIST Neural Network (from scratch, NumPy only)")
    st.write(
        """
        Demo of a handwritten digit classifier.

        The neural network is implemented **from scratch** using only NumPy
        (no PyTorch / no TensorFlow) and trained on the **MNIST** dataset.
        """
    )

    with st.spinner("Loading data and training the model (first time only)..."):
        model, X_test, y_test = train_model()
    st.success("Model trained and ready!")

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Step 1: Pick a test image")
        if st.button("🔀 Random digit"):
            st.session_state.idx = int(np.random.randint(0, len(X_test)))

        idx = st.session_state.idx
        img = X_test[idx].reshape(28, 28)
        true_label = int(y_test[idx])

        st.image(
            img,
            caption=f"True label: {true_label}",
            width=200,
            clamp=True,
        )

    with col_right:
        st.subheader("Step 2: Let the model predict")

        if st.button("🔮 Predict digit"):
            probs, _ = model.forward(X_test[idx:idx + 1])
            probs = probs.flatten()
            pred_label = int(np.argmax(probs))

            st.markdown(f"### Predicted digit: `{pred_label}`")

            if pred_label == true_label:
                st.success("Correct prediction 🎉")
            else:
                st.error("Wrong prediction ❌")

            st.write("**Class probabilities:**")
            prob_dict = {str(i): float(probs[i]) for i in range(10)}
            st.bar_chart(prob_dict)

        else:
            st.info("Click **Predict digit** to run the model on this image.")


if __name__ == "__main__":
    main()
