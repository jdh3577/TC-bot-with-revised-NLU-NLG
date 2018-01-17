#coding:utf-8
import pickle

class Config(object):
    data_dir = 'deep_dialog/nlg/Data_v1/'
    vec_file = 'Data/vec5.txt'
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 10 #gradient clipping
    num_layers = 2

    with open(data_dir+"sentences.pickle", 'rb') as f:
        sentences = pickle.load(f)

    lenn = []
    for i in range(len(sentences)):
        lenn.append(len(sentences[i].split()))

    num_steps = max(lenn)+1 #this value is one more than max number of words in sentence
    hidden_size = 20
    word_embedding_size = 30
    max_epoch = 30
    max_max_epoch = 1000
    keep_prob = 0.5 #The probability that each element is kept through dropout layer
    lr_decay = 1.0
    batch_size = 16
    vocab_size = 199
    keyword_min_count = 1
    save_freq = 10 #The step (counted by the number of iterations) at which the model is saved to hard disk.
    model_path = 'deep_dialog/nlg//Model_v1/Model_News' #the path of model that need to save or load
    
    # parameter for generation
    len_of_generation = num_steps #The number of characters by generated
    save_time = 1000 #load save_time saved models
    is_sample = False #true means using sample, if not using argmax
    BeamSize = 2
