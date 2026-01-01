import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Sigmoid Activation Function")

st.write(
    "This web application visualises the Sigmoid activation function. " \
    "It maps input values into a range between 0 and 1, making it suitable for binary classification problems where outputs represent probabilities."
)

x = np.linspace(-10, 10, 400)
y = 1 / (1 + np.exp(-x))

plt.figure()
plt.plot(x, y)
plt.xlabel("Input (x)")
plt.ylabel("Sigmoid(x)")
plt.title("Sigmoid Function")
plt.grid(True)

st.pyplot(plt)
