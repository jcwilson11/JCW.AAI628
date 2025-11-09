# %%
from keras.datasets import imdb 
(train_data, train_labels),(test_data, test_labels) = imdb.load_data( num_words=10000)

# %% [markdown]
# 

# %%
train_data[0]

# %%
train_labels[0:20]

# %%
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# %%


# %%
x_train = vectorize_sequences(train_data) 
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# %%
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# %%
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %%

from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])




# %%
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])


# %%
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# %%
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


# %%
history_dict = history.history
history_dict.keys()


# %%
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
epochs = range(1, len(acc_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %%
# 1 hidden layer of 16 units
model_1layer = models.Sequential()
model_1layer.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model_1layer.add(layers.Dense(1, activation='sigmoid'))

model_1layer.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['acc'])

history_1layer = model_1layer.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

test_loss_1, test_acc_1 = model_1layer.evaluate(x_test, y_test)
print("1-layer model test accuracy:", test_acc_1)



# %%
# 3 hidden layers of 16 units each
model_3layer = models.Sequential()
model_3layer.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model_3layer.add(layers.Dense(16, activation='relu'))
model_3layer.add(layers.Dense(16, activation='relu'))
model_3layer.add(layers.Dense(1, activation='sigmoid'))

model_3layer.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                     loss='binary_crossentropy',
                     metrics=['acc'])

history_3layer = model_3layer.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

test_loss_3, test_acc_3 = model_3layer.evaluate(x_test, y_test)
print("3-layer model test accuracy:", test_acc_3)


# %%
model_32 = models.Sequential()
model_32.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model_32.add(layers.Dense(32, activation='relu'))
model_32.add(layers.Dense(1, activation='sigmoid'))

model_32.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['acc'])

history_32 = model_32.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

test_loss_32, test_acc_32 = model_32.evaluate(x_test, y_test)
print("32-unit model test accuracy:", test_acc_32)

# %%
model_64 = models.Sequential()
model_64.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model_64.add(layers.Dense(64, activation='relu'))
model_64.add(layers.Dense(1, activation='sigmoid'))

model_64.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['acc'])

history_64 = model_64.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

test_loss_64, test_acc_64 = model_64.evaluate(x_test, y_test)
print("64-unit model test accuracy:", test_acc_64)

# %%
model_128 = models.Sequential()
model_128.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
model_128.add(layers.Dense(128, activation='relu'))
model_128.add(layers.Dense(1, activation='sigmoid'))

model_128.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['acc'])

history_128 = model_128.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

test_loss_128, test_acc_128 = model_128.evaluate(x_test, y_test)
print("128-unit model test accuracy:", test_acc_128)

# %%
model_256 = models.Sequential()
model_256.add(layers.Dense(256, activation='relu', input_shape=(10000,)))
model_256.add(layers.Dense(256, activation='relu'))
model_256.add(layers.Dense(1, activation='sigmoid'))

model_256.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['acc'])

history_256 = model_256.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

test_loss_256, test_acc_256 = model_256.evaluate(x_test, y_test)
print("256-unit model test accuracy:", test_acc_256)

# %%
# Baseline architecture but with MSE loss
model_mse = models.Sequential()
model_mse.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model_mse.add(layers.Dense(16, activation='relu'))
model_mse.add(layers.Dense(1, activation='sigmoid'))

model_mse.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                  loss='mse',
                  metrics=['acc'])

history_mse = model_mse.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

test_loss_mse, test_acc_mse = model_mse.evaluate(x_test, y_test)
print("MSE loss model test accuracy:", test_acc_mse) #better than baseline


# %%
# tanh
model_tanh = models.Sequential()
model_tanh.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))
model_tanh.add(layers.Dense(16, activation='tanh'))
model_tanh.add(layers.Dense(1, activation='sigmoid'))

model_tanh.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                   loss='binary_crossentropy',
                   metrics=['acc'])

history_tanh = model_tanh.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

test_loss_tanh, test_acc_tanh = model_tanh.evaluate(x_test, y_test)
print("tanh model test accuracy:", test_acc_tanh) #slightly worse than baseline

# %%
# Baseline architecture but with MSE loss and 256 units
model_mse = models.Sequential()
model_mse.add(layers.Dense(256, activation='relu', input_shape=(10000,)))
model_mse.add(layers.Dense(256, activation='relu'))
model_mse.add(layers.Dense(1, activation='sigmoid'))

model_mse.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                  loss='mse',
                  metrics=['acc'])

history_mse = model_mse.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

test_loss_mse, test_acc_mse = model_mse.evaluate(x_test, y_test)
print("MSE loss + 256 model test accuracy:", test_acc_mse) #better than baseline


# %%
# Baseline architecture but with MSE loss and 256 units and 3 layers
model_mse = models.Sequential()
model_mse.add(layers.Dense(256, activation='relu', input_shape=(10000,)))
model_mse.add(layers.Dense(256, activation='relu'))
model_mse.add(layers.Dense(256, activation='relu'))
model_mse.add(layers.Dense(1, activation='sigmoid'))

model_mse.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                  loss='mse',
                  metrics=['acc'])

history_mse = model_mse.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

test_loss_mse, test_acc_mse = model_mse.evaluate(x_test, y_test)
print("MSE loss + 256 + 3 layers model test accuracy:", test_acc_mse) #better than baseline



