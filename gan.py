# ---------------------- Import Libraries ----------------------
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# Set random seed for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ---------------------- Load Dataset ----------------------
data = pd.read_csv("updated_features_output.csv")

# Extract only the URLs
data = data.dropna(subset=['URL'])  # Drop rows with missing URLs
urls = data['URL'].astype(str).tolist()  # Convert to list of strings

# ---------------------- Preprocessing ----------------------
# Define a character set for encoding URLs
chars = sorted(set("".join(urls)))  # Unique characters in dataset
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

max_url_length = max(len(url) for url in urls)  # Find longest URL

# Encode URLs into numerical sequences
def encode_url(url):
    encoded = [char_to_idx[c] for c in url] + [0] * (max_url_length - len(url))  # Pad with zeros
    return encoded

# Apply encoding
X_train = np.array([encode_url(url) for url in urls])

# ---------------------- Build GAN Models ----------------------

# Generator: LSTM-based model that generates synthetic URLs
def build_generator(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(max_length * vocab_size, activation="softmax"))  # Output probabilities for each character
    return model

# Discriminator: Classifies URLs as real (1) or fake (0)
def build_discriminator(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=max_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(1, activation="sigmoid"))  # Binary classification
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# GAN: Combines Generator and Discriminator
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze discriminator while training generator
    gan_input = Input(shape=(max_url_length,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    return gan

# ---------------------- Initialize Models ----------------------
vocab_size = len(chars) + 1  # +1 for padding
generator = build_generator(vocab_size, max_url_length)
discriminator = build_discriminator(vocab_size, max_url_length)
gan = build_gan(generator, discriminator)

# Compile GAN
gan.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.5), loss="binary_crossentropy")

# ---------------------- Training the GAN ----------------------
num_epochs = 500
batch_size = 16
half_batch = int(batch_size / 2)

for epoch in range(num_epochs):
    # Select real URLs
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    X_real = X_train[idx]
    y_real = np.ones((half_batch, 1))  # Real labels

    # Generate synthetic URLs
    noise = np.random.randint(0, vocab_size, (half_batch, max_url_length))
    X_fake = generator.predict(noise)
    y_fake = np.zeros((half_batch, 1))  # Fake labels

    # Train Discriminator
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(X_real, y_real)
    d_loss_fake = discriminator.train_on_batch(X_fake, y_fake)

    # Train Generator via GAN
    noise = np.random.randint(0, vocab_size, (batch_size, max_url_length))
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))  # Fool discriminator

    # Print losses every 10 epochs
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch}, D-Loss Real: {d_loss_real[0]:.4f}, D-Loss Fake: {d_loss_fake[0]:.4f}, G-Loss: {g_loss[0]:.4f}")

# ---------------------- Generate Synthetic URLs ----------------------
num_synthetic_urls = 1000  # Adjust as needed
generated_sequences = generator.predict(np.random.randint(0, vocab_size, (num_synthetic_urls, max_url_length)))

# Convert back to text
def decode_url(encoded_seq):
    return "".join([idx_to_char[int(i)] for i in encoded_seq if i in idx_to_char])

synthetic_urls = [decode_url(seq) for seq in generated_sequences]

# Save synthetic URLs to file
synthetic_df = pd.DataFrame({"URL": synthetic_urls})
synthetic_df.to_csv("synthetic_urls.csv", index=False)
print("Synthetic URLs saved as 'synthetic_urls.csv'.")

# ---------------------- Next Steps ----------------------
# Run feature extraction on synthetic_urls.csv using your existing code