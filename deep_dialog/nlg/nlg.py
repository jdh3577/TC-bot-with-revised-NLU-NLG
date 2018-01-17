'''
Created on Oct 17, 2016

--dia_act_nl_pairs.v6.json: agt and usr have their own NL.


@author: xiul
'''

import pickle
import copy, argparse, json
import numpy as np
import smart_open

from .utils import *

from deep_dialog import dialog_config
from deep_dialog.nlg.lstm_decoder_tanh import lstm_decoder_tanh

from deep_dialog.nlg.sc_lstm import *
from operator import itemgetter

class nlg:
    def __init__(self):
        # kwd_voc = pickle.load(open('kwd_voc.pkl','rb'))
        gen_config.key_words_voc_size = 45

        self.word_vec = pickle.load(open(gen_config.data_dir+'word_vec.pkl', 'rb'))
        self.vocab = pickle.load(open(gen_config.data_dir+'word_voc.pkl', 'rb'))

        self.word_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_word = {i: ch for i, ch in enumerate(self.vocab)}
        # keyword_to_idx = { ch:i for i,ch in enumerate(kwd_voc) }

        gen_config.vocab_size = len(self.vocab)
        self.beam_size = gen_config.BeamSize

        self.session = tf.Session(config=config_tf)

        #with tf.Graph().as_default(),  as session:

        gen_config.batch_size = 1
        gen_config.num_steps = 1

        initializer = tf.random_uniform_initializer(-gen_config.init_scale,
                                                    gen_config.init_scale)
        with tf.variable_scope("nlg_model", reuse=None, initializer=initializer):
            self.mtest = SC_LSTM_Model(is_training=False, config=gen_config)

        #self.session.run(tf.initialize_all_variables())



        print('model loading ...')

        all_vars = tf.all_variables()
        model_nlg_vars = [k for k in all_vars if k.name.startswith("nlg_model")]

        model_saver = tf.train.Saver(model_nlg_vars)
        model_saver.restore(self.session, gen_config.model_path + '--%d' % gen_config.save_time)
        print('Done!')

        pass
    
    def post_process(self, pred_template, slot_val_dict, slot_dict):
        """ post_process to fill the slot in the template sentence """

        '''
        suffix = "_PLACEHOLDER"

        for slot in slot_val_dict.keys():
            slot_vals = slot_val_dict[slot]
            slot_placeholder = '$' + slot + '$'
            if slot == 'result' or slot == 'numberofpeople': continue
            if slot_vals == dialog_config.NO_VALUE_MATCH: continue
            tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
            sentence = tmp_sentence
                
        if 'numberofpeople' in slot_val_dict.keys():
            slot_vals = slot_val_dict['numberofpeople']
            slot_placeholder = '$' + 'numberofpeople' + '$'
            tmp_sentence = sentence.replace(slot_placeholder, slot_vals, 1)
            sentence = tmp_sentence
    
        for slot in slot_dict:
            slot_placeholder = '$' + slot + '$'
            tmp_sentence = sentence.replace(slot_placeholder, '')
            sentence = tmp_sentence
        '''

        sentence_tmp = pred_template.split()

        slot_sentence_index = []
        sort_slot_index = []
        sort_slot = ['$date$', '$starttime$', '$city$', '$state$', '$theater$', '$moviename$', '$video_format$', '$numberofpeople$']

        slot_to_idx = {ch: i for i, ch in enumerate(sort_slot)}
        idx_to_slot = {i: ch for i, ch in enumerate(sort_slot)}
        end_index = -1
        for i in range(len(sentence_tmp)):
            if sentence_tmp[i] in sort_slot:
                slot_sentence_index.append(i)
                sort_slot_index.append(slot_to_idx[sentence_tmp[i]])
            if sentence_tmp[i] == 'END' and end_index < 0:
                end_index = i

        sort_slot_index.sort()
        #sort_slot_index, slot_sentence_index = zip(*sorted(zip(sort_slot_index, slot_sentence_index)))

        date_index = -1
        space_index = -1

        for i in range(1,-1,-1):
            if sort_slot[i] in sentence_tmp:
                date_index = i
                break

        for i in range(4,1,-1):
            if sort_slot[i] in sentence_tmp:
                space_index = i
                break


        for i in range(len(sort_slot_index)):
            if date_index >-1 and sort_slot_index[i] == date_index:
                sentence_tmp[slot_sentence_index[i]] = idx_to_slot[sort_slot_index[i]]
            elif space_index >-1 and sort_slot_index[i] == space_index:
                sentence_tmp[slot_sentence_index[i]] = idx_to_slot[sort_slot_index[i]]
            else:
                sentence_tmp[slot_sentence_index[i]] = idx_to_slot[sort_slot_index[i]]

        if end_index != -1:
            sentence = ' '.join(sentence_tmp[:end_index])
        else:
            sentence = ' '.join(sentence_tmp)

        counter = 0
        for slot in list(slot_val_dict.keys()):
            slot_val = slot_val_dict[slot]
            if slot_val == dialog_config.NO_VALUE_MATCH:
                sentence = dialog_config.NO_VALUE_MATCH
                break
            elif slot_val == dialog_config.I_DO_NOT_CARE:
                counter += 1
                sentence = sentence.replace('$' + slot + '$', '', 1)
                continue

            sentence = sentence.replace('$' + slot + '$', slot_val, 1)

        if counter > 0 and counter == len(slot_val_dict):
            sentence = dialog_config.I_DO_NOT_CARE

        return sentence

    
    def convert_diaact_to_nl(self, dia_act, turn_msg):
        """ Convert Dia_Act into NL: Rule + Model """
        
        sentence = ""
        boolean_in = False

        # remove I do not care slot in task(complete)
        if dia_act['diaact'] == 'inform' and 'taskcomplete' in dia_act['inform_slots'].keys() and dia_act['inform_slots']['taskcomplete'] != dialog_config.NO_VALUE_MATCH:
            inform_slot_set = list(dia_act['inform_slots'].keys())
            for slot in inform_slot_set:
                if dia_act['inform_slots'][slot] == dialog_config.I_DO_NOT_CARE: del dia_act['inform_slots'][slot]
        
        if dia_act['diaact'] in self.diaact_nl_pairs['dia_acts'].keys():
            for ele in self.diaact_nl_pairs['dia_acts'][dia_act['diaact']]:
                if set(ele['inform_slots']) == set(dia_act['inform_slots'].keys()) and set(ele['request_slots']) == set(dia_act['request_slots'].keys()):
                    sentence = self.post_process(ele['nl'][turn_msg], dia_act['inform_slots'], slot_dict)
                    #sentence = self.diaact_to_nl_slot_filling(dia_act, ele['nl'][turn_msg])
                    boolean_in = True
                    break
        
        if dia_act['diaact'] == 'inform' and 'taskcomplete' in dia_act['inform_slots'].keys() and dia_act['inform_slots']['taskcomplete'] == dialog_config.NO_VALUE_MATCH:
            sentence = "Oh sorry, there is no ticket available."
        
        if boolean_in == False:
            sentence = self.translate_diaact(dia_act)

        return sentence
        
        
    def translate_diaact(self, dia_act):
        """ prepare the diaact into vector representation, and generate the sentence by Model """
        '''
        word_dict = self.word_dict
        template_word_dict = self.template_word_dict
        act_dict = self.act_dict
        slot_dict = self.slot_dict
        inverse_word_dict = self.inverse_word_dict
    
        act_rep = np.zeros((1, len(act_dict)))
        act_rep[0, act_dict[dia_act['diaact']]] = 1.0
    
        slot_rep_bit = 2
        slot_rep = np.zeros((1, len(slot_dict)*slot_rep_bit)) 
    
        suffix = "_PLACEHOLDER"
        if self.params['dia_slot_val'] == 2 or self.params['dia_slot_val'] == 3:
            word_rep = np.zeros((1, len(template_word_dict)))
            words = np.zeros((1, len(template_word_dict)))
            words[0, template_word_dict['s_o_s']] = 1.0
        else:
            word_rep = np.zeros((1, len(word_dict)))
            words = np.zeros((1, len(word_dict)))
            words[0, word_dict['s_o_s']] = 1.0
    
        for slot in dia_act['inform_slots'].keys():
            slot_index = slot_dict[slot]
            slot_rep[0, slot_index*slot_rep_bit] = 1.0
        
            for slot_val in dia_act['inform_slots'][slot]:
                if self.params['dia_slot_val'] == 2:
                    slot_placeholder = slot + suffix
                    if slot_placeholder in template_word_dict.keys():
                        word_rep[0, template_word_dict[slot_placeholder]] = 1.0
                elif self.params['dia_slot_val'] == 1:
                    if slot_val in word_dict.keys():
                        word_rep[0, word_dict[slot_val]] = 1.0
                    
        for slot in dia_act['request_slots'].keys():
            slot_index = slot_dict[slot]
            slot_rep[0, slot_index*slot_rep_bit + 1] = 1.0
    
        if self.params['dia_slot_val'] == 0 or self.params['dia_slot_val'] == 3:
            final_representation = np.hstack([act_rep, slot_rep])
        else: # dia_slot_val = 1, 2
            final_representation = np.hstack([act_rep, slot_rep, word_rep])
    
        dia_act_rep = {}
        dia_act_rep['diaact'] = final_representation
        dia_act_rep['words'] = words
    
        #pred_ys, pred_words = nlg_model['model'].forward(inverse_word_dict, dia_act_rep, nlg_model['params'], predict_model=True)
        pred_ys, pred_words = self.model.beam_forward(inverse_word_dict, dia_act_rep, self.params, predict_model=True)
        pred_sentence = ' '.join(pred_words[:-1])
        sentence = self.post_process(pred_sentence, dia_act['inform_slots'], slot_dict)
        '''

        #with tf.Graph().as_default(), tf.Session(config=config_tf) as session:

        tmp = []
        beams = [(0.0, [self.idx_to_word[1]], self.idx_to_word[1])]

        tmp = np.zeros(gen_config.key_words_voc_size)

        tmp[act_to_idx[dia_act['diaact']]] = 1.0

        for slot in dia_act['inform_slots'].keys():
            tmp[len(act_dict) + slot_to_idx[slot]] = 1.0
        for slot in dia_act['request_slots'].keys():
            tmp[len(act_dict) + len(slot_dict) + slot_to_idx[slot]] = 1.0

        _input_words = np.array([tmp], dtype=np.float32)
        test_data = np.int32([1])
        prob, _state, _last_output, _sc_vec = run_epoch(self.session, self.mtest, test_data, sc_vec=_input_words, flag=0)
        y1 = np.log(1e-20 + prob.reshape(-1))
        if gen_config.is_sample:
            try:
                top_indices = np.random.choice(gen_config.vocab_size, self.beam_size, replace=False, p=prob.reshape(-1))
            except:
                top_indices = np.random.choice(gen_config.vocab_size, self.beam_size, replace=True, p=prob.reshape(-1))
        else:
            top_indices = np.argsort(-y1)
        b = beams[0]
        beam_candidates = []
        for i in range(self.beam_size):
            wordix = top_indices[i]
            beam_candidates.append((b[0] + y1[wordix], b[1] + [self.idx_to_word[wordix]], wordix, _state, _last_output, _sc_vec))

        beam_candidates.sort(key=lambda x: x[0], reverse=True)  # decreasing order
        beams = beam_candidates[:self.beam_size]  # truncate to get new beams

        for xy in range(gen_config.len_of_generation - 1):
            beam_candidates = []
            for b in beams:
                test_data = np.int32(b[2])
                prob, _state, _last_output, _sc_vec = run_epoch(self.session, self.mtest, test_data, b[3], flag=1,last_output=b[4], sc_vec=b[5])
                y1 = np.log(1e-20 + prob.reshape(-1))
                if gen_config.is_sample:
                    try:
                        top_indices = np.random.choice(gen_config.vocab_size, self.beam_size, replace=False,p=prob.reshape(-1))
                    except:
                        top_indices = np.random.choice(gen_config.vocab_size, self.beam_size, replace=True,p=prob.reshape(-1))
                else:
                    top_indices = np.argsort(-y1)
                # beam_candidates.append(b)
                for i in range(self.beam_size):
                    wordix = top_indices[i]
                    beam_candidates.append((b[0] + y1[wordix], b[1] + [self.idx_to_word[wordix]], wordix, _state, _last_output, _sc_vec))
            beam_candidates.sort(key=lambda x: x[0], reverse=True)  # decreasing order
            beams = beam_candidates[:self.beam_size]  # truncate to get new beams

        pred_sentence = ' '.join(beams[0][1][1:])
        sentence = self.post_process(pred_sentence, dia_act['inform_slots'], slot_dict)

        return sentence
    
    
    def load_nlg_model(self, model_path):
        """ load the trained NLG model """  
        
        model_params = pickle.load(open(model_path, 'rb'), encoding='iso-8859-1')
    
        hidden_size = model_params['model']['Wd'].shape[0]
        output_size = model_params['model']['Wd'].shape[1]
    
        if model_params['params']['model'] == 'lstm_tanh': # lstm_tanh
            diaact_input_size = model_params['model']['Wah'].shape[0]
            input_size = model_params['model']['WLSTM'].shape[0] - hidden_size - 1
            rnnmodel = lstm_decoder_tanh(diaact_input_size, input_size, hidden_size, output_size)
        
        rnnmodel.model = copy.deepcopy(model_params['model'])
        model_params['params']['beam_size'] = dialog_config.nlg_beam_size
        
        self.model = rnnmodel
        self.word_dict = copy.deepcopy(model_params['word_dict'])
        self.template_word_dict = copy.deepcopy(model_params['template_word_dict'])
        self.slot_dict = copy.deepcopy(model_params['slot_dict'])
        self.act_dict = copy.deepcopy(model_params['act_dict'])
        self.inverse_word_dict = {self.template_word_dict[k]:k for k in self.template_word_dict.keys()}
        self.params = copy.deepcopy(model_params['params'])
        
    
    def diaact_to_nl_slot_filling(self, dia_act, template_sentence):
        """ Replace the slots with its values """
        
        sentence = template_sentence
        counter = 0
        for slot in list(dia_act['inform_slots'].keys()):
            slot_val = dia_act['inform_slots'][slot]
            if slot_val == dialog_config.NO_VALUE_MATCH:
                sentence = slot + " is not available!"
                break
            elif slot_val == dialog_config.I_DO_NOT_CARE:
                counter += 1
                sentence = sentence.replace('$'+slot+'$', '', 1)
                continue
            
            sentence = sentence.replace('$'+slot+'$', slot_val, 1)
        
        if counter > 0 and counter == len(dia_act['inform_slots']):
            sentence = dialog_config.I_DO_NOT_CARE
        
        return sentence
    
    
    def load_predefine_act_nl_pairs(self, path):
        """ Load some pre-defined Dia_Act&NL Pairs from file """

        with open(path, 'rb') as f:
            self.diaact_nl_pairs = pickle.load(f)

        '''
        self.diaact_nl_pairs = json.load(smart_open.smart_open(path, 'r'))
        
        for key in list(self.diaact_nl_pairs['dia_acts'].keys()):
            for ele in self.diaact_nl_pairs['dia_acts'][key]:
                ele['nl']['usr'] = ele['nl']['usr'].encode('utf-8') # encode issue
                ele['nl']['agt'] = ele['nl']['agt'].encode('utf-8') # encode issue
        '''


def main(params):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    params = vars(args)

    print ("User Simulator Parameters:")
    print (json.dumps(params, indent=2))

    main(params)
