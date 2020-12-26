import numpy as np
import re
import string
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, GRU, Embedding

from flask import Flask, request, render_template
app = Flask(__name__)

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

emb_dim = 64
x_vocab = 12799
y_vocab = 26515
dec_units = 128
enc_units = 128
att_units = 10
enc_inp_length = 47
dec_inp_length = 49

with open('eng_tokenizer.pickle', 'rb') as handle:
    eng_tokenizer = pickle.load(handle)

with open('ita_tokenizer.pickle', 'rb') as handle:
    ita_tokenizer = pickle.load(handle)
    
def decontractions(phrase):
    '''Performs decontractions in the doc'''

    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"couldn\'t", "could not", phrase)
    phrase = re.sub(r"shouldn\'t", "should not", phrase)
    phrase = re.sub(r"wouldn\'t", "would not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
        
    return phrase

def preprocess(line):
    '''Function does simple preprocessing steps such as converting to lowercase, removing punctuations and
       separating the english and italian sentences'''
    line = ''.join(e for e in line if e.isdigit() == False)
    line = re.sub(' +', ' ', line)
    line = line.lower()
    line = decontractions(line)
    temp1 = []
    for char in line:
        if char in string.punctuation:
            continue
        else:
            temp1.append(char)
    # removing links
    if re.findall(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', line):
        line = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', line)
    # removing html if any
    if re.findall('<.*?>',line):
        line = re.sub('<.*?>','',line)    
    return line

def get_model():
    
    class Encoder(tf.keras.layers.Layer):
        def __init__(self, vocab_size, embedding_dim, input_length, enc_units):
            super(Encoder, self).__init__()
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.input_length = input_length
            self.enc_units= enc_units
            self.gru_output = 0
            self.state_h=0

        def build(self, input_shape):
            self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                                       mask_zero=True, name="embedding_layer_encoder")
            self.gru = GRU(self.enc_units, return_state=True, return_sequences=True, name="Encoder_GRU")

        def call(self, input_sentences, training=True):

            input_embed = self.embedding(input_sentences)

            self.gru_output, self.gru_state_h = self.gru(input_embed)

            return self.gru_output, self.gru_state_h

        def get_states(self):

            return self.state_h
        
    class OneStepDecoder(tf.keras.layers.Layer):
        def __init__(self, vocab_size, emb_dim, att_units, dec_units):
            super(OneStepDecoder, self).__init__()
            self.vocab_size = vocab_size
            self.emb_dim = emb_dim
            self.att_units = att_units
            self.dec_units = dec_units

        def build(self, input_shape):
            self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.emb_dim, input_length=49, mask_zero=True,
                                       name="embedding_layer_decoder")
            self.gru = GRU(self.dec_units, return_sequences=True, return_state=True, name="Decoder_GRU")
            self.fc = Dense(self.vocab_size)

            self.V = Dense(1)
            self.W = Dense(self.att_units)
            self.U = Dense(self.att_units)

        def call(self, dec_input, hidden_state, enc_output):

            hidden_with_time = tf.expand_dims(hidden_state, 1)

            attention_weights = self.V(tf.nn.tanh(self.U(enc_output) + self.W(hidden_with_time)))

            attention_weights = tf.nn.softmax(attention_weights, 1)

            context_vector = attention_weights * enc_output

            context_vector = tf.reduce_sum(context_vector, axis=1)


            x = self.embedding(dec_input)
            x = tf.concat([tf.expand_dims(context_vector, axis=1),x], axis=-1)
            output, h_state = self.gru(x, initial_state = hidden_state)

            output = tf.reshape(output, (-1, output.shape[2]))

            x = self.fc(output)

            return x, h_state, attention_weights
    
    class Decoder(tf.keras.layers.Layer):
    
        def __init__(self, embedding_dim, vocab_size, input_length, dec_units, att_units):
            super(Decoder, self).__init__()
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.input_length = input_length
            self.dec_units = dec_units
            self.att_units = att_units
            self.onestep_decoder = OneStepDecoder(self.vocab_size, self.embedding_dim, self.att_units, self.dec_units)
        @tf.function    
        def call(self, dec_input, hidden_state, enc_output):
            all_outputs = tf.TensorArray(tf.float32, dec_input.shape[1], name='output_arrays')

            for timestep in range(dec_input.shape[1]):

                output, hidden_state, attention_weights = self.onestep_decoder(dec_input[:, timestep:timestep+1], 
                                                                               hidden_state, enc_output)

                all_outputs = all_outputs.write(timestep, output)

            all_outputs = tf.transpose(all_outputs.stack(), [1,0,2])
            return all_outputs
    
    class Attention_Model(tf.keras.Model):
        def __init__(self, embedding_dim, x_vocab, y_vocab, dec_units, enc_units, enc_inp_length, dec_inp_length, att_units):
            super(Attention_Model, self).__init__()
            self.encoder = Encoder(x_vocab, embedding_dim, enc_inp_length, enc_units)
            self.decoder = Decoder(embedding_dim, y_vocab, dec_inp_length,dec_units, att_units)

        def call(self, data):
            enc_input, dec_input = data[0], data[1]

            enc_output, enc_state = self.encoder(enc_input)

            dec_output = self.decoder(dec_input, enc_state, enc_output)

            return dec_output
        
    model1 = Attention_Model(emb_dim, x_vocab, y_vocab, dec_units, enc_units, enc_inp_length, dec_inp_length, att_units)
    model1.build([(None, 47),(None, 49)])
    model1.load_weights('model_log_m1_fit2_09.h5')
    
    return model1

@app.route('/')
def home():
    return render_template("index3.html")

@app.route('/predict', methods=['POST'])
def predict():

    model1 = get_model()
    inputs = request.form.get("english")
    inputs = preprocess(inputs)
    in_ = len(inputs.split()) - 1
    inputs = [inputs]
    inputs = np.array(eng_tokenizer.texts_to_sequences(inputs))
    inputs = pad_sequences(inputs, 47, padding='post', truncating='post')
    enc_output, enc_state = model1.layers[0](inputs)
    input_state = enc_state
    pred = []
    cur_vec = np.array([ita_tokenizer.word_index['startseq']]).reshape(-1,1)
    att_weights = np.zeros((in_, 49))
    for i in range(49):
        inf_output, input_state, attention_weights = model1.layers[1].onestep_decoder(cur_vec, input_state, enc_output)
        cur_vec = np.reshape(np.argmax(inf_output), (1, 1))
        if cur_vec[0][0] != 0:
            pred.append(cur_vec)
        else:
            break
    final = ' '.join([ita_tokenizer.index_word[e[0][0]] for e in pred if e[0][0] != 0 and e[0][0] != 2])
    
    
    return render_template("index3.html", prediction_text='Translates to :- \n {}'.format(final))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    #app.run()