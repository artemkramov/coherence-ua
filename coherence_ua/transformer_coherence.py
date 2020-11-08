from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM
from .models.utils import *
import tensorflow_datasets as tfds
import pathlib
from os.path import join
import ufal.udpipe


# Multihead attention class
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# Encoder layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)

        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


# Encoder object that consists of embedding, positional encoding and a number of Encoder layers
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)

        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# Coherence model that performs the estimation of the coherence of the given text
class CoherenceModel:

    # Deep learning model to estimate the coherence of a clique
    model = None

    # Number of sentences that each clique contains
    clique_length = 3

    def __init__(self):

        # Set current folder in order to load all necessary models
        self.current_folder = join(pathlib.Path(__file__).parent.absolute(), "models")

        # Load a tokenizer to transform sentences into vectors
        self.tokenizer_sentence = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
            join(self.current_folder, 'vocab'))

        # Load a Transformer-based neural network model
        self.load_model()

        # Load a Ukrainian dependency model to preprocess an input text
        self.language_model = UniversalDependencyModel(join(self.current_folder, 'ukrainian-iu-ud-2.3-181115.udpipe'))

    # Init a Transformer-based neural network model and load corresponding weights
    def load_model(self):
        num_layers = 4
        dff = 512
        num_heads = 8
        vocab_size = self.tokenizer_sentence.vocab_size + 1  # plus zero
        dropout_rate = 0.1
        embedding_dim = 512
        self.model = TransformerCoherence(num_layers, embedding_dim, num_heads, dff, vocab_size, dropout_rate)
        self.model.load_weights(join(self.current_folder, "weights", "chk-2-0.0917-0.8678"))

    # Preprocess input text in order to pass it to the neural network model
    def preprocess_text(self, text):
        MAX_LENGTH = 30

        # Split the text into sentences
        sentences = self.language_model.tokenize(text)
        document = []

        # Loop through each sentence, split them into word, perform lemmatization
        for s in sentences:

            # Parse each sentence to retrieve lemma
            self.language_model.tag(s)
            self.language_model.parse(s)

            # Collect all lemmas into one list
            i = 0
            words = []
            while i < len(s.words):
                words.append(s.words[i].lemma)
                i += 1
            document.append(" ".join(words[1:]))

        # Convert sentences into the set of numbers according to the loaded tokenizer
        sequences = tf.keras.preprocessing.sequence.pad_sequences(
            [self.tokenizer_sentence.encode(sentence) for sentence in document], maxlen=MAX_LENGTH, value=0)
        return sequences

    # Get probabilities for the set of the cliques of the given text
    def get_prediction_series(self, text):

        # Convert the text into sequences
        sequences = self.preprocess_text(text)

        # Init samples list
        x_samples = []

        # Set an initial counter
        counter = self.clique_length - 1

        # Don't allow to process short texts
        if self.clique_length > len(sequences):
            return np.array([0])

        # Loop through the text
        while counter < len(sequences):

            # Form a clique
            x_sample = []
            for i in range(counter - self.clique_length, counter):
                x_sample.append(sequences[i])
            counter += 1
            x_samples.append(x_sample)

        # Prepare input data for the NN model
        inputs = np.array(x_samples)

        # Form encoding padding masks for the inputs
        enc_padding_masks = [create_padding_mask(inputs[:, 0, :]), create_padding_mask(inputs[:, 1, :]),
                             create_padding_mask(inputs[:, 2, :])]

        # Return the output of the model
        return self.model(inputs, False, enc_padding_masks)

    # Calculate the coherence of the text as the product of the probabilities of cliques
    def evaluate_coherence_as_product(self, text):
        return tf.math.reduce_prod(self.get_prediction_series(text)).numpy()

    # Calculate the coherence of the text as the ratio of a number of coherent cliques over all cliques
    # according to the corresponding threshold
    def evaluate_coherence_using_threshold(self, text, threshold=0.25):
        predictions = self.get_prediction_series(text)
        predictions = [1 if output > threshold else 0 for output in predictions]
        return np.sum(predictions) / len(predictions)

# Deep NN model that is based on the Transformer architecture
class TransformerCoherence(Model):

    def __init__(self, num_layers, embedding_dim, num_heads, dff, vocab_size, dropout_rate, **kwargs):
        super(TransformerCoherence, self).__init__(**kwargs)

        self.encoder = Encoder(num_layers, embedding_dim, num_heads, dff, vocab_size, vocab_size, dropout_rate)
        self.lstm1 = LSTM(embedding_dim, activation='relu')
        self.dense_common = Dense(512, activation='relu')
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, inputs, training, enc_padding_masks):
        z1 = self.dense_common(self.lstm1(self.encoder(inputs[:, 0, :], training, enc_padding_masks[0])))
        z2 = self.dense_common(self.lstm1(self.encoder(inputs[:, 1, :], training, enc_padding_masks[1])))
        z3 = self.dense_common(self.lstm1(self.encoder(inputs[:, 2, :], training, enc_padding_masks[2])))

        concat_layer = tf.concat([z1, z2, z3], axis=-1)
        x = self.dense1(concat_layer)

        return self.dense2(x)


class UniversalDependencyModel:
    # udpipe compiled model
    model = None

    def __init__(self, path):
        # Load model by the given path
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load model by the given path: %s" % path)

    def parse(self, sentence):
        self.model.parse(sentence, self.model.DEFAULT)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output
