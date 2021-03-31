# n_way classification
# hidden size
def make_lstm_config(n_way, h_size):
    print('make lstm config')

    config = [
        ('linear', [256, 768]),  # character encoder
        ('gru', [h_size, 256]),  # GRU encoder-decoder
        #### classifier
        ('linear', [n_way, h_size]),  # classifier
    ]
    return config


def make_bilstm_config(n_way, h_size):
    print('make bilstm config')

    config = [
        ('linear', [256, 768]),  # character encoder
        ('gru_bi', [h_size, 256]),  # GRU encoder-decoder
        #### classifier
        ('linear', [n_way, h_size]),  # classifier
    ]
    return config


def make_lstm_multi_task_config(n_way, h_size, n_topic):
    print('make lstm multitask config')

    config = [
        ('linear', [256, 768]),  # character encoder
        ('gru', [h_size, 256]),  # GRU encoder-decoder
        ('linear', [h_size, h_size]),
        ('relu', [True]),
        #### classifier
        ('linear', [n_way, h_size]),  # classifier for rumor classification
        ('linear', [n_topic, h_size]),  # classifier for topic classification
    ]
    return config


def make_bilstm_multi_task_config(n_way, h_size, n_topic):
    print('make lstm multitask config')

    config = [
        ('linear', [256, 768]),  # character encoder
        ('gru_bi', [h_size, 256]),  # GRU encoder-decoder
        ('linear', [h_size, h_size]),
        ('relu', [True]),
        #### classifier
        ('linear', [n_way, h_size]),  # classifier for rumor classification
        ('linear', [n_topic, h_size]),  # classifier for topic classification
    ]
    return config


def make_bilstm_multi_layer_multi_task_config(n_way, h_size, n_layer, n_topic):
    print('make lstm multitask config')

    config = [
        ('linear', [256, 768]),  # character encoder
        ('gru_bi_multilayer', [h_size, 256, n_layer]),  # GRU encoder-decoder
        ('linear', [h_size, h_size]),
        ('relu', [True]),
        #### classifier
        ('linear', [n_way, h_size]),  # classifier for rumor classification
        ('linear', [n_topic, h_size]),  # classifier for topic classification
    ]
    return config


###########################################################
## baselines

def make_fcn_baseline_config(n_way, h_size):
    print('make lstm config')

    config = [
        ('linear', [256, 768]),  # character encoder
        #### classifier
        ('linear', [n_way, 256]),  # classifier
    ]
    return config


def make_rnn_baseline_config(n_way, h_size):
    print('make lstm config')

    config = [
        ('linear', [256, 768]),  # character encoder
        ('gru_bi_multilayer', [h_size, 256, 2]),  # GRU encoder-decoder
        #### classifier
        ('linear', [n_way, h_size]),  # classifier
    ]
    return config
