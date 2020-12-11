# Adopted from Tensorflow with some modifications

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np

from MultiHeadAttention import MultiHeadAttention
from EncoderLayer import EncoderLayer
from DecoderLayer import DecoderLayer
from Encoder import Encoder
from Decoder import Decoder
from Transformer import Transformer
from CustomSchedule import CustomSchedule

import urllib
from speech2text import Speech2Text


# Input
input1 = urllib.request.urlopen('https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en')
input2 = urllib.request.urlopen('https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de')

temp1, temp2 = [], []
train_temp_1, train_temp_2 = [], []
for line in input1:
    temp1.append(line.decode())
for line in input2:
    temp2.append(line.decode())
for i in range(0, len(temp1)):
    val1 = tf.convert_to_tensor(temp1[i], dtype=tf.string)
    val2 = tf.convert_to_tensor(temp2[i], dtype=tf.string)
    train_temp_1.append(val1)
    train_temp_2.append(val2)
train_temp_1 = tf.data.Dataset.from_tensor_slices(train_temp_1)
train_temp_2 = tf.data.Dataset.from_tensor_slices(train_temp_2)
train_examples = tf.data.Dataset.zip((train_temp_1, train_temp_2))

input1 = urllib.request.urlopen('https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.en')
input2 = urllib.request.urlopen('https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.de')
temp1, temp2 = [], []
val_temp_1, val_temp_2 = [], []
for line in input1:
    temp1.append(line.decode())
for line in input2:
    temp2.append(line.decode())
for i in range(0, len(temp1)):
    val1 = tf.convert_to_tensor(temp1[i], dtype=tf.string)
    val2 = tf.convert_to_tensor(temp2[i], dtype=tf.string)
    val_temp_1.append(val1)
    val_temp_2.append(val2)
val_temp_1 = tf.data.Dataset.from_tensor_slices(val_temp_1)
val_temp_2 = tf.data.Dataset.from_tensor_slices(val_temp_2)
val_examples = tf.data.Dataset.zip((val_temp_1, val_temp_2))


# Prepare data
print('Preparing Language 1')
tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((en.numpy() for en, de in train_examples), target_vocab_size=2**13)
print('Preparing Language 2')
tokenizer_de = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((de.numpy() for en, de in train_examples), target_vocab_size=2**13)


# Initialization
BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 40

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_en.vocab_size + 2
target_vocab_size = tokenizer_de.vocab_size + 2
dropout_rate = 0.1

EPOCHS = 5

total_time = time.time()


# Defining functions
def encode(lang1, lang2):
    lang1 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang1.numpy()) + [tokenizer_en.vocab_size+1]
    lang2 = [tokenizer_de.vocab_size] + tokenizer_de.encode(lang2.numpy()) + [tokenizer_de.vocab_size+1]
  return lang1, lang2


def tf_encode(en, de):
    result_en, result_de = tf.py_function(encode, [en, de], [tf.int64, tf.int64])
    result_en.set_shape([None])
    result_de.set_shape([None])
    return result_en, result_de


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                 True,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))


def evaluate(inp_sentence):
    start_token = [tokenizer_en.vocab_size]
    end_token = [tokenizer_en.vocab_size + 1]

    inp_sentence = start_token + tokenizer_en.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    decoder_input = [tokenizer_de.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == tokenizer_de.vocab_size+1:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def translate(sentence):
    start = time.time()
    result, attention_weights = evaluate(sentence)
    predicted_sentence = tokenizer_de.decode([i for i in result if i < tokenizer_de.vocab_size])
    total_time = total_time + time.time() - start
    return predicted_sentence


train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes = ([None],[None]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes = ([None],[None]))


# Initializing model
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

checkpoint_path = "./checkpoints/train/"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')


# Start Training
for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)
        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


# Overall evaluation (for scores & time ...)
test_set = []
file = open('translation.txt', 'w')
for de, pt in val_examples:
    test_set.append(de.numpy().decode('utf-8'))
for example in test_set:
    result = translate(example)
    file.write('Input: ' + str(example) + '\n')
    file.write('Predicted translation: ' + str(result) + '\n\n')
file.close()
print('Average translation time: ' + str(total_time/len(test_set)))


# Speech-Text evaluation
file = open('translation2.txt', 'w')
recognizer = Speech2Text()
for i in range(0, 5):
    input_text = recognizer.stt('./audio/' + str(i) + '.wav')
    result = translate(input_text)
    file.write('Input: ' + str(example) + '\n')
    file.write('Predicted translation: ' + str(result) + '\n\n')
file.close()
