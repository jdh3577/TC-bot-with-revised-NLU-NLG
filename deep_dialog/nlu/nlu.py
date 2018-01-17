'''
Created on Jul 13, 2016

@author: xiul
'''

import pickle
import copy
import numpy as np

from deep_dialog.nlu.lstm import lstm
from deep_dialog.nlu.bi_lstm import biLSTM



import tensorflow as tf

from deep_dialog.nlu import data_utils
from deep_dialog.nlu import multi_task_model


tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 128, "Size of the word embedding")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 10000, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "data\\TC_bot_dataset", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "model_tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 300,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 12000,
                            "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Max size of test set.")
tf.app.flags.DEFINE_boolean("use_attention", True,
                            "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 50,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "dropout keep cell input and output prob.")
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True,
                            "Use birectional RNN")
tf.app.flags.DEFINE_string("task", "joint", "Options: joint; intent; tagging")
FLAGS = tf.app.flags.FLAGS

#
# if FLAGS.max_sequence_length == 0:
#     print ('Please indicate max sequence length. Exit')
#     exit()
#
# if FLAGS.task is None:
#     print ('Please indicate task to run. Available options: intent; tagging; joint')
#     exit()

task = dict({'intent':0, 'tagging':0, 'joint':0})

task['intent'] = 1
task['tagging'] = 1
task['joint'] = 1





class nlu:
    def __init__(self):
        self.input_target = dict()

        self.sess = tf.Session()

        vocab_path, tag_vocab_path, label_vocab_path = data_utils.prepare_multi_task_data(FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)

        self.vocab, self.rev_vocab = data_utils.initialize_vocabulary(vocab_path)
        self.tag_vocab, self.rev_tag_vocab = data_utils.initialize_vocabulary(tag_vocab_path)
        self.label_vocab, self.rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)

        self.model, self.model_test = self.create_model(self.sess, len(self.vocab), len(self.tag_vocab), len(self.label_vocab))

        pass

    def create_model(self, session, source_vocab_size, target_vocab_size, label_vocab_size):

        _buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]
        """Create model and initialize or load parameters in session."""
        with tf.variable_scope("nlu_model", reuse=None):
            model_train = multi_task_model.MultiTaskModel(
                source_vocab_size, target_vocab_size, label_vocab_size, _buckets,
                FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
                dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
                forward_only=False,
                use_attention=FLAGS.use_attention,
                bidirectional_rnn=FLAGS.bidirectional_rnn,
                task=task)
        with tf.variable_scope("nlu_model", reuse=True):
            model_test = multi_task_model.MultiTaskModel(
                source_vocab_size, target_vocab_size, label_vocab_size, _buckets,
                FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
                dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
                forward_only=True,
                use_attention=FLAGS.use_attention,
                bidirectional_rnn=FLAGS.bidirectional_rnn,
                task=task)

        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

        if ckpt:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model_train.saver.restore(session, ckpt.model_checkpoint_path)
            print('Done!')

        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
        return model_train, model_test

    def run_valid_test(self, tmp_annot):  # mode: Eval, Test

        UNK_ID = data_utils.UNK_ID_dict['with_padding']

        token_ids = data_utils.sentence_to_token_ids(tmp_annot, self.vocab, UNK_ID, data_utils.naive_tokenizer, True)

        # Run evals on development/test set and print the accuracy.
        word_list = list()
        hyp_tag_list = list()
        hyp_label_list = list()

        result = ["O"]
        correct_count = 0
        accuracy = 0.0
        tagging_eval_result = dict()


        encoder_inputs, tags, tag_weights, sequence_length, labels = self.model_test.get_one(token_ids)
        tagging_logits = []
        classification_logits = []

        _, step_loss, tagging_logits, classification_logits = self.model_test.joint_step(self.sess, encoder_inputs, tags, tag_weights, labels, sequence_length, 0, True)

        hyp_label = None
        if task['tagging'] == 1:
            word_list.append([self.rev_vocab[x[0]] for x in encoder_inputs[:sequence_length[0]]])
            hyp_tag_list.append([self.rev_tag_vocab[np.argmax(x)] for x in tagging_logits[:sequence_length[0]]])
            result = result + [self.rev_tag_vocab[np.argmax(x)] for x in tagging_logits[:sequence_length[0]]]

        if task['intent'] == 1:
            hyp_label = np.argmax(classification_logits[0], 0)
            hyp_label_list.append(self.rev_label_vocab[hyp_label])
            result.append(self.rev_label_vocab[hyp_label])

        return result

    def generate_dia_act(self, annot):
        """ generate the Dia-Act with NLU model """
        
        if len(annot) > 0:
            tmp_annot = annot.strip('.').strip('?').strip(',').strip('!')

            rep = self.parse_str_to_vector(tmp_annot)
            Ys, cache = self.model.fwdPass(rep, self.params, predict_model=True) # default: True

            maxes = np.amax(Ys, axis=1, keepdims=True)
            e = np.exp(Ys - maxes) # for numerical stability shift into good numerical range
            probs = e/np.sum(e, axis=1, keepdims=True)
            if np.all(np.isnan(probs)): probs = np.zeros(probs.shape)

            # special handling with intent label
            for tag_id in self.inverse_tag_dict.keys():
                if self.inverse_tag_dict[tag_id].startswith('B-') or self.inverse_tag_dict[tag_id].startswith('I-') or self.inverse_tag_dict[tag_id] == 'O':
                    probs[-1][tag_id] = 0

            pred_words_indices = np.nanargmax(probs, axis=1)
            pred_tags = [self.inverse_tag_dict[index] for index in pred_words_indices]



            pred_tags = self.run_valid_test(tmp_annot)

            diaact = self.parse_nlu_to_diaact(pred_tags, tmp_annot)

            #print("intent : "+ ' '.join(str(x) for x in pred_tags) + " / diact : "+diaact['diaact'])

            return diaact
        else:
            return None

    
    def load_nlu_model(self, model_path):
        """ load the trained NLU model """  
        
        model_params = pickle.load(open(model_path, 'rb'), encoding='latin-1')
    
        hidden_size = model_params['model']['Wd'].shape[0]
        output_size = model_params['model']['Wd'].shape[1]
    
        if model_params['params']['model'] == 'lstm': # lstm_
            input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
            rnnmodel = lstm(input_size, hidden_size, output_size)
        elif model_params['params']['model'] == 'bi_lstm': # bi_lstm
            input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
            rnnmodel = biLSTM(input_size, hidden_size, output_size)
           
        rnnmodel.model = copy.deepcopy(model_params['model'])
        
        self.model = rnnmodel
        self.word_dict = copy.deepcopy(model_params['word_dict'])
        self.slot_dict = copy.deepcopy(model_params['slot_dict'])
        self.act_dict = copy.deepcopy(model_params['act_dict'])
        self.tag_set = copy.deepcopy(model_params['tag_set'])
        self.params = copy.deepcopy(model_params['params'])

        self.inverse_tag_dict = {self.tag_set[k]: k for k in self.tag_set.keys()}

    def parse_str_to_vector(self, string):
        """ Parse string into vector representations """
        
        tmp = 'BOS ' + string + ' EOS'
        words = tmp.lower().split()
        
        vecs = np.zeros((len(words), len(self.word_dict)))
        for w_index, w in enumerate(words):
            if w.endswith(',') or w.endswith('?'):
                w = w[0:-1]
            if w in self.word_dict.keys():
                vecs[w_index][self.word_dict[w]] = 1
            else: vecs[w_index][self.word_dict['unk']] = 1
        
        rep = {}
        rep['word_vectors'] = vecs
        rep['raw_seq'] = string
        return rep

    def parse_nlu_to_diaact(self, nlu_vector, string):
        """ Parse BIO and Intent into Dia-Act """
        
        tmp = 'BOS ' + string + ' EOS'
        words = tmp.lower().split(' ')
    
        diaact = {}
        diaact['diaact'] = "inform"
        diaact['request_slots'] = {}
        diaact['inform_slots'] = {}
        
        intent = nlu_vector[-1]
        index = 1
        pre_tag = nlu_vector[0]
        pre_tag_index = 0
    
        slot_val_dict = {}
    
        while index<(len(nlu_vector)-1): # except last Intent tag
            cur_tag = nlu_vector[index]
            if cur_tag == 'O' and pre_tag.startswith('B-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
            elif cur_tag.startswith('B-') and pre_tag.startswith('B-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
            elif cur_tag.startswith('B-') and pre_tag.startswith('I-'):
                if cur_tag.split('-')[1] != pre_tag.split('-')[1]:           
                    slot = pre_tag.split('-')[1]
                    slot_val_str = ' '.join(words[pre_tag_index:index])
                    slot_val_dict[slot] = slot_val_str
            elif cur_tag == 'O' and pre_tag.startswith('I-'):
                slot = pre_tag.split('-')[1]
                slot_val_str = ' '.join(words[pre_tag_index:index])
                slot_val_dict[slot] = slot_val_str
               
            if cur_tag.startswith('B-'):
                pre_tag_index = index
        
            pre_tag = cur_tag
            index += 1
    
        if cur_tag.startswith('B-') or cur_tag.startswith('I-'):
            slot = cur_tag.split('-')[1]
            slot_val_str = ' '.join(words[pre_tag_index:-1])
            slot_val_dict[slot] = slot_val_str

        if intent != 'null':
            arr = intent.split('+')
            diaact['diaact'] = arr[0]
            diaact['request_slots'] = {}
            for ele in arr[1:]:

                if ele == 'description':
                    aa = 1

                #request_slots.append(ele)
                diaact['request_slots'][ele] = 'UNK'
        
        diaact['inform_slots'] = slot_val_dict
         
        # add rule here
        for slot in diaact['inform_slots'].keys():
            slot_val = diaact['inform_slots'][slot]
            if slot_val.startswith('bos'): 
                slot_val = slot_val.replace('bos', '', 1)
                diaact['inform_slots'][slot] = slot_val.strip(' ')
        
        self.refine_diaact_by_rules(diaact)
        return diaact

    def refine_diaact_by_rules(self, diaact):
        """ refine the dia_act by rules """
        
        # rule for taskcomplete
        if 'request_slots' in diaact.keys():
            if 'taskcomplete' in diaact['request_slots'].keys():
                del diaact['request_slots']['taskcomplete']
                diaact['inform_slots']['taskcomplete'] = 'PLACEHOLDER'
        
            # rule for request
            if len(diaact['request_slots'])>0: diaact['diaact'] = 'request'

    def diaact_penny_string(self, dia_act):
        """ Convert the Dia-Act into penny string """
        
        penny_str = ""
        penny_str = dia_act['diaact'] + "("
        for slot in dia_act['request_slots'].keys():
            penny_str += slot + ";"
    
        for slot in dia_act['inform_slots'].keys():
            slot_val_str = slot + "="
            if len(dia_act['inform_slots'][slot]) == 1:
                slot_val_str += dia_act['inform_slots'][slot][0]
            else:
                slot_val_str += "{"
                for slot_val in dia_act['inform_slots'][slot]:
                    slot_val_str += slot_val + "#"
                slot_val_str = slot_val_str[:-1]
                slot_val_str += "}"
            penny_str += slot_val_str + ";"
    
        if penny_str[-1] == ";": penny_str = penny_str[:-1]
        penny_str += ")"
        return penny_str