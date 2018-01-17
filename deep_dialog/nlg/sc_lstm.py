
from deep_dialog.nlg import Config
from deep_dialog.nlg.SC_LSTM_Model import SC_LSTM
from deep_dialog.nlg.SC_LSTM_Model import SC_MultiRNNCell
from deep_dialog.nlg.SC_LSTM_Model import SC_DropoutWrapper


import tensorflow as tf



class SC_LSTM_Model(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.size = size = config.hidden_size
        vocab_size = config.vocab_size
        key_words_voc_size = config.key_words_voc_size

        alpha = tf.constant(0.5)

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])  # 声明输入变量x, y
        self._input_word = tf.placeholder(tf.float32, [batch_size, key_words_voc_size])
        self._mask = tf.placeholder(tf.float32, [batch_size, num_steps])

        LSTM_cell = SC_LSTM(key_words_voc_size, size, forget_bias=0.0, state_is_tuple=False)
        if is_training and config.keep_prob < 1:
            LSTM_cell = SC_DropoutWrapper(
                LSTM_cell, output_keep_prob=config.keep_prob)
        cell = SC_MultiRNNCell([LSTM_cell] * config.num_layers, state_is_tuple=False)

        self._initial_state = cell.zero_state(batch_size, tf.float32)
        self._init_output = tf.zeros([batch_size, size * config.num_layers], tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable('word_embedding', [vocab_size, config.word_embedding_size], trainable=True)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        sc_vec = self._input_word

        outputs = []
        output_state = self._init_output
        state = self._initial_state

        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                with tf.variable_scope("RNN_sentence"):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()

                    sc_wr = tf.get_variable('sc_wr', [config.word_embedding_size, key_words_voc_size])
                    res_wr = tf.matmul(inputs[:, time_step, :], sc_wr)

                    res_hr = tf.zeros_like(res_wr, dtype=tf.float32)
                    for layer_id in range(config.num_layers):
                        sc_hr = tf.get_variable('sc_hr_%d' % layer_id, [size, key_words_voc_size])
                        res_hr += alpha * tf.matmul(tf.slice(output_state, [0, size * layer_id], [-1, size]), sc_hr)
                    r_t = tf.sigmoid(res_wr + res_hr)
                    sc_vec = r_t * sc_vec

                    (cell_output, state, cell_outputs) = cell(inputs[:, time_step, :], state, sc_vec)
                    outputs.append(cell_outputs)
                    output_state = cell_outputs

            self._sc_vec = sc_vec
            self._end_output = output_state

        try:
            output = tf.reshape(tf.concat(1, outputs), [-1, size * config.num_layers])
        except:
            output = tf.reshape(tf.concat(outputs, 1), [-1, size * config.num_layers])
        softmax_w = tf.get_variable("softmax_w", [size * config.num_layers, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b

        self._final_state = state
        self._prob = tf.nn.softmax(logits)

        return

    @property
    def input_data(self):
        return self._input_data

    @property
    def end_output(self):
        return self._end_output

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state


def run_epoch(session, m, data, state=None, sc_vec=None, flag=1, last_output=None):
    """Runs the model on the given data."""
    x = data.reshape((1, 1))
    if flag == 0:
        prob, _state, _last_output, _sc_vec = session.run([m._prob, m.final_state, m.end_output, m._sc_vec],
                                                          {m.input_data: x,
                                                           m._input_word: sc_vec})
    else:
        prob, _state, _last_output, _sc_vec = session.run([m._prob, m.final_state, m.end_output, m._sc_vec],
                                                          {m.input_data: x,
                                                           m._input_word: sc_vec,
                                                           m.initial_state: state,
                                                           m._init_output: last_output})
    return prob, _state, _last_output, _sc_vec


config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True

gen_config = Config.Config()

act_dict = ['request', 'inform', 'confirm_question', 'confirm_answer', 'greeting', 'closing', 'multiple_choice',
            'thanks', 'welcome', 'deny', 'not_sure']
slot_dict = ['actor', 'actress', 'city', 'closing', 'date', 'genre', 'greeting', 'moviename', 'numberofpeople',
             'taskcomplete', 'starttime', 'state', 'theater', 'video_format', 'result', 'ticket','other']

act_to_idx = { ch:i for i,ch in enumerate(act_dict) }
idx_to_act = { i:ch for i,ch in enumerate(act_dict) }

slot_to_idx = { ch:i for i,ch in enumerate(slot_dict) }
idx_to_slot = { i:ch for i,ch in enumerate(slot_dict) }