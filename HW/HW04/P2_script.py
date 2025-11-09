# %% [markdown]
# ### Q1: Import / Load

# %%
# Step 1: Import libraries and set random seeds
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

label_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

x_train = x_train / 255.0
x_test = x_test / 255.0

# Visualize a few samples
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(label_names[y_train[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### one hot encode
# 

# %%
# Step 1 (continued): Flatten and normalize, then one-hot encode
X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
X_test  = X_test.reshape(-1, 28*28).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)


# %% [markdown]
# ### Define and compile model

# %%
def create_model():
    model = Sequential([
        layers.Input(shape=(784,)),         
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = create_model()
model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# %% [markdown]
# ### Train model

# %%
# Step 4: Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=128,
    verbose=1
)


# %% [markdown]
# ### plot

# %%
# Step 5: Evaluate test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc*100:.2f}%")

# Plot accuracy and loss
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# # Q2
# Q2. For a given set of images with titles and without titles, which data set will have a better image classification performance? Explain and justify your answer.
# 
# 
# The dataset with titles (labeled images) will lead to significantly better classification performance, because supervised learning relies on labeled data to compute error, adjust model weights, and improve predictive accuracy.
# Unlabeled images cannot train a classifier well without additional labeling or semi-supervised techniques.


