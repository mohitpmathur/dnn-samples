"""
Create word embeddings and classify imdb movie reviews
"""

import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras import Sequential

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = "post"
oov_token = "<OOV>"

# Load IMDB dataset
imdb, info = tfds.load("imdb_reviews", with_info=True,as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']
print("Dataset loaded successfully !!")

train_sentences = []
train_labels = []

test_sentences = []
test_labels = []

for s,l in train_data:
    train_sentences.append(s.numpy().decode('utf8'))
    train_labels.append(l.numpy())

for s,l in test_data:
    test_sentences.append(s.numpy().decode('utf8'))
    test_labels.append(l.numpy())

train_labels_final = np.array(train_labels)
test_labels_final = np.array(test_labels)
print("Dataset processed !")

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)
print("Training sequences & padded sequences created ...")

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length)
print("Testing sequences & padded sequences created ...")

reverse_word_index  = tokenizer.index_word

def decode_review(text):
    return " ".join([reverse_word_index.get(i, '?') for i in text])

# print(decode_review(train_padded[3]))
# print(decode_review(train_sequences[3]))

print("Starting to build the model ...")
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Flatten(),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(f"Model summary: {model.summary()}")

num_epochs = 10
history = model.fit(train_padded,
                    train_labels_final,
                    epochs=num_epochs,
                    validation_data=(test_padded, test_labels_final),
                    )
print("Fit completed")

# Get weights on embedding layer
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

import io

# create vector & metadata files
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

# if running in Google Colab, download files
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')


