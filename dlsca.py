from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam, RMSprop # type: ignore
#from tensorflow.keras.optimizers import * # type: ignore
#from tensorflow.keras.layers import Dense, Sequential, Conv1D, BatchNormalization, AveragePooling1D, Flatten, Dropout # type: ignore
from tensorflow.keras.layers import * # type: ignore
from tensorflow.keras.models import * # type: ignore
from tensorflow.keras import callbacks # type: ignore
import numpy as np
import scipy.stats as stats
import os
from io import StringIO
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import json

debug_flag = False
traces_path = './dlsca_traces/profiling_traces/no_avgs/'
attack_traces_path = './dlsca_traces/attack_traces/no_avgs/'
# Test with successful CPA traces
#attack_traces_path = './dlsca_traces/attack_traces/key_16_kr_8/'
figures_path = './Figures/'
params_file = 'dlsca_params.txt'


# Class that contains utilities for use throughout the document
class Auxiliar:
    
    # Find the covariance between two 1D lists (x and y).
    # Note that var(x) = cov(x, x)
    def cov(x, y):
        return np.cov(x, y)[0][1]
    
    # Returns the AES Substitution Box Value given an input Byte Value
    sbox=(
        0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
        0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
        0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
        0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
        0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
        0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
        0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
        0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
        0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
        0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
        0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
        0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
        0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
        0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
        0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
        0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16)
    
    # Returns the number of set bits for a given byte
    hw = [bin(x).count("1") for x in range(256)]

    #### AUXILIAR FUNCTIONS #############################################################################

    # Returns the KEY of the file
    def getKeyName(filename):
        return filename.split("_")[-1].split(".")[0]

    # Returns the BLOCK SIZE of the file
    def getBlockSize(filename):
        return filename.split("_")[-4]

    # Create a learning rate scheduler callback.
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    )
    # Create an early stopping callback.
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    # Flattens a given matrix into a single list (row values remain adjacent)
    def my_flatten(matrix):
        #print(matrix)
        return [item for sublist in matrix for item in sublist]


    # Loads .npy arrays and keeps each sublist unaveraged
    def loadNPY(path):
        traces_dict = {}
        for key_dir in os.listdir(path):
            if os.path.isdir(os.path.join(path,key_dir)):
                key_int = int(key_dir.split("_")[1])
                if (key_int not in traces_dict.keys()):
                    traces_dict[key_int] = {"power": [], "plaintexts": [], "energy": [], "clock_cycles": []}
                list_of_npy = sorted(os.listdir(os.path.join(path, key_dir))) #sorted to ensure plaintexts aren't first and we get a shape to transform our plaintext list
                for sets in range(len(list_of_npy) // len(traces_dict[key_int].keys())):
                    for npy_file in list_of_npy:
                        if sets == int(npy_file.split(".")[0].split("_")[-1]): #ensure we iterate sets per order
                            if "clock_cycles" in npy_file:
                                shape_matrix = np.load(os.path.join(path,key_dir,npy_file), allow_pickle=True)
                                traces_dict[key_int]["clock_cycles"].extend(Auxiliar.my_flatten(np.load(os.path.join(path,key_dir,npy_file), allow_pickle=True)))
                            elif "power" in npy_file:
                                traces_dict[key_int]["power"].extend(Auxiliar.my_flatten(np.load(os.path.join(path,key_dir,npy_file), allow_pickle=True)))
                            elif "energy" in npy_file:
                                traces_dict[key_int]["energy"].extend(Auxiliar.my_flatten(np.load(os.path.join(path,key_dir,npy_file), allow_pickle=True)))
                            elif "plaintexts" in npy_file:
                                plaintexts_list = np.load(os.path.join(path,key_dir,npy_file), allow_pickle=True)
                                plaintexts_list_repeated = []
                                for plaintext_iter in range(len(plaintexts_list)):
                                    for i in range(len(shape_matrix[plaintext_iter])):
                                        plaintexts_list_repeated.append(plaintexts_list[plaintext_iter])
                                traces_dict[key_int]["plaintexts"].extend(plaintexts_list_repeated)
                if debug_flag:
                    print(f"Length of power k{key_int}", len(traces_dict[key_int]["power"]))
                    print(f"Length of plaintexts k{key_int}", len(traces_dict[key_int]["plaintexts"]))
                    print(f"Length of energy k{key_int}", len(traces_dict[key_int]["energy"]))
                    print(f"Length of clock_cycles k{key_int}", len(traces_dict[key_int]["clock_cycles"]))
        return traces_dict
    
    # Loads .npy arrays and averages each sublist into a single datapoint
    def loadNPY_avg(path):
        traces_dict = {}
        for key_dir in os.listdir(path):
            if os.path.isdir(os.path.join(path,key_dir)):
                key_int = int(key_dir.split("_")[1])
                if (key_int not in traces_dict.keys()):
                    traces_dict[key_int] = {"power": [], "plaintexts": [], "energy": [], "clock_cycles": []}
                list_of_npy = sorted(os.listdir(os.path.join(path, key_dir))) #sorted to ensure plaintexts aren't first and we get a shape to transform our plaintext list
                for sets in range(len(list_of_npy) // len(traces_dict[key_int].keys())):
                    for npy_file in list_of_npy:
                        if sets == int(npy_file.split(".")[0].split("_")[-1]): #ensure we iterate sets per order
                            if "clock_cycles" in npy_file:
                                clock_cycles_matrix = np.load(os.path.join(path,key_dir,npy_file), allow_pickle=True)
                                for list in clock_cycles_matrix:
                                    traces_dict[key_int]["clock_cycles"].append(np.mean(list))
                            elif "power" in npy_file:
                                power_matrix = np.load(os.path.join(path,key_dir,npy_file), allow_pickle=True)
                                for list in power_matrix:
                                    traces_dict[key_int]["power"].append(np.mean(list))
                            elif "energy" in npy_file:
                                energy_matrix = np.load(os.path.join(path,key_dir,npy_file), allow_pickle=True)
                                for list in energy_matrix:
                                    traces_dict[key_int]["energy"].append(np.mean(list))
                            elif "plaintexts" in npy_file:
                                plaintexts_list = np.load(os.path.join(path,key_dir,npy_file), allow_pickle=True)
                                traces_dict[key_int]["plaintexts"].extend(plaintexts_list)
                if debug_flag:
                    print(f"Length of power k{key_int}", len(traces_dict[key_int]["power"]))
                    print(f"Length of plaintexts k{key_int}", len(traces_dict[key_int]["plaintexts"]))
                    print(f"Length of energy k{key_int}", len(traces_dict[key_int]["energy"]))
                    print(f"Length of clock_cycles k{key_int}", len(traces_dict[key_int]["clock_cycles"]))
        return traces_dict
    
    # Prints HW class of each sub key byte guess
    def analyze_hws():
        traces_dict = Auxiliar.loadNPY_avg(traces_path)
        keys = traces_dict.keys()
        hws = [[] for _ in range(9)]
        for key in keys:
            hws[Auxiliar.hw[key]].append(key)
        
        for hw in range(9):
            print(f"Number of keys with HW{hw}: {len(hws[hw])}")

        hws_supposed = [[] for _ in range(9)]
        for key in range(256):
            hws_supposed[Auxiliar.hw[key]].append(key)

        for hw in range(9):
            print(f"Number of supposed keys with HW{hw}: {len(hws_supposed[hw])}")

    # Decapsulates parameters from a dictionary for better usability
    def decapsulate_parameters_dict(parameters_dict):
        layer_size = parameters_dict["layer_size"]
        model_epochs = parameters_dict["model_epochs"]
        model_batch_size = parameters_dict["model_batch_size"]
        callback = parameters_dict["callback"]
        input_feature_option = parameters_dict["input_feature_option"]
        model = parameters_dict["model"]
        return layer_size, model_epochs, model_batch_size, callback, input_feature_option, model


# Class that contains the definition of the used DL models, a CNN and a MLP 
class Models():
    
    # AISY - Deep Learning-based Framework for Side Channel Analysis - url={https://eprint.iacr.org/2021/357}
    def aisy_mlp(classes, number_of_samples, layer_size):
        model = Sequential(name="aisy_mlp")
        model.add(Dense(layer_size, activation='selu', input_shape=(number_of_samples,)))
        model.add(Dense(layer_size, activation='selu'))
        model.add(Dense(layer_size, activation='selu'))
        model.add(Dense(layer_size, activation='selu'))
        model.add(Dense(classes, activation='softmax'))
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'],)
        summary_str = StringIO()
        model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
        summary_str = summary_str.getvalue()
        return model, summary_str
    
    # AISY - Deep Learning-based Framework for Side Channel Analysis - url={https://eprint.iacr.org/2021/357}
    def aisy_cnn(classes, number_of_samples, layer_size):
        model = Sequential(name="aisy_cnn")
        model.add(Conv1D(filters=layer_size, kernel_size=1, strides=1, activation='relu', padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(AveragePooling1D(pool_size=1, strides=1))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        summary_str = StringIO()
        model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
        summary_str = summary_str.getvalue()
        return model, summary_str
    
    # UNUSED Ranking Loss: Maximizing the Success Rate in Deep Learning Side-Channel Analysis - url={https://tches.iacr.org/index.php/TCHES/article/view/8726}, DOI={10.46586/tches.v2021.i1.25-55}, 
    def rank_loss_cnn(classes, input_size, learning_rate=0.00001, alpha_value=10.0,  rkl_loss=True):
        
        # Personal design
        input_shape = (input_size,1)
        img_input = Input(shape=input_shape, dtype='float32')

        # 1st convolutional block
        x = Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
        x = BatchNormalization()(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        
        x = Flatten(name='flatten')(x)

        # Classification layer
        x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
        x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
        
        # Logits layer
        score_layer = Dense(classes, activation=None, name='score')(x)
        predictions = Activation('softmax')(score_layer)

        # Create model
        inputs = img_input
        model = Model(inputs, predictions, name='rank_loss_cnn')
        optimizer = Adam(lr=learning_rate)

        if(rkl_loss==True):
            #model.compile(loss=self.ranking_loss_optimized(score_layer),optimizer=optimizer, metrics=['accuracy'])
            model.compile(loss=Loss_Functions.loss_sca(score_layer),optimizer=optimizer, metrics=['accuracy'])
        else:
            model.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics=['accuracy'])

        return model

# UNUSED - Class that contains unimplemented custom DL-SCA Loss functions 
class Loss_Functions():

    # Naive implementation
    # Ranking Loss: Maximizing the Success Rate in Deep Learning Side-Channel Analysis - url={https://tches.iacr.org/index.php/TCHES/article/view/8726}, DOI={10.46586/tches.v2021.i1.25-55}, 
    def loss_sca(self, score_vector):

        # Rank loss function
        def ranking_loss_sca(y_true, y_pred):
            alpha_value = 10.0
            nb_class = self.number_of_keys
            #alpha = K.constant(alpha_value, dtype='float32')
            alpha = tf.constant(alpha_value, dtype='float32')

            # Batch_size initialization
            #y_true_int = K.cast(y_true, dtype='int32')
            y_true_int = tf.cast(y_true, dtype='int32')
            #batch_s = K.cast(K.shape(y_true_int)[0],dtype='int32')
            batch_s = tf.cast(tf.shape(y_true_int)[0],dtype='int32')

            # Indexing the training set (range_value = (?,))
            #range_value = K.arange(0, batch_s, dtype='int64')
            range_value = tf.keras.backend.arange(0, batch_s, dtype='int64')

            # Get rank and scores associated with the secret key (rank_sk = (?,))
            values_topk_logits, indices_topk_logits = tf.nn.top_k(score_vector, k=nb_class, sorted=True) # values_topk_logits = shape(?, nb_class) ; indices_topk_logits = shape(?, nb_class)
            rank_sk = tf.where(tf.equal(tf.cast(indices_topk_logits, dtype='int64'), tf.reshape(tf.argmax(y_true_int), [tf.shape(tf.argmax(y_true_int))[0], 1])))[:,1] + 1 # Index of the correct output among all the hypotheses (shape(?,))
            score_sk = tf.gather_nd(values_topk_logits, tf.keras.backend.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]), tf.reshape(rank_sk-1, [tf.shape(rank_sk)[0], 1])])) # Score of the secret key (shape(?,))

            # Ranking Loss Initialization
            loss_rank = 0

            for i in range(nb_class):

                # Score for each key hypothesis (s_i_shape=(?,))
                s_i = tf.gather_nd(values_topk_logits, tf.keras.backend.concatenate([tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]), i*tf.ones([tf.shape(values_topk_logits)[0], 1], dtype='int64')]))

                # Indicator function identifying when (i == secret key)
                indicator_function = tf.ones(batch_s) - (tf.cast(tf.equal(rank_sk-1, i), dtype='float32') * tf.ones(batch_s))

                # Logistic loss computation
                #logistic_loss = K.log(1 + K.exp(- alpha * (score_sk - s_i)))/K.log(2.0)
                logistic_loss = tf.keras.backend.log(1 + tf.keras.backend.exp(- alpha * (score_sk - s_i)))/tf.keras.backend.log(2.0)
                

                # Ranking Loss computation
                loss_rank = tf.reduce_sum((indicator_function * logistic_loss))+loss_rank

            return loss_rank/(tf.cast(batch_s, dtype='float32'))

        #Return the ranking loss function
        return ranking_loss_sca


# Class that contains different options for input designs to feed to the architecture
class Diff_Input_Features():

    # NOT TESTED - energy consumption // clock cycles // plaintext input -----> Key Byte Value
    def energy_cycles_input___key_id(traces_dict, max_size_test = 0):
        y_label = []
        x_train = []
        sorted_keys = sorted(traces_dict.keys())
        if max_size_test == 0:
            max_size_test = len(traces_dict[sorted_keys[0]]["power"])
        for key in sorted_keys:
            for i in range(max_size_test):
                x_train_iter = np.zeros(3)
                x_train_iter[0] = traces_dict[key]["energy"][i]
                x_train_iter[1] = traces_dict[key]["clock_cycles"][i]
                x_train_iter[2] = traces_dict[key]["plaintexts"][i]
                x_train.append(x_train_iter)
                label = np.zeros(len(sorted_keys))
                label[sorted_keys.index(key)] = 1
                y_label.append(label)
        x_train = np.array(x_train)
        y_label = np.array(y_label)
        scaler = MinMaxScaler(feature_range=(0,1))
        x_train = scaler.fit_transform(x_train)
        assert len(x_train) == len(y_label), f"x_train and y_label have different lengths!!!, {len(x_train)}, {len(y_label)}"
        return x_train, y_label

    # energy consumption // clock cycles // plaintext input // S-Box HW -----> Key Byte Value
    def energy_cycles_input_HW___key_id(traces_dict, max_size_test = 0):
        y_label = []
        x_train = []
        sorted_keys = sorted(traces_dict.keys())
        if max_size_test == 0:
            max_size_test = len(traces_dict[sorted_keys[0]]["power"])
        for key in sorted_keys:
            for i in range(max_size_test):
                x_train_iter = np.zeros(4)
                x_train_iter[0] = traces_dict[key]["energy"][i]
                x_train_iter[1] = traces_dict[key]["clock_cycles"][i]
                x_train_iter[2] = traces_dict[key]["plaintexts"][i]
                x_train_iter[3] = Auxiliar.hw[Auxiliar.sbox[(traces_dict[key]["plaintexts"][i] ^ key)]]
                x_train.append(x_train_iter)
                label = np.zeros(len(sorted_keys))
                label[sorted_keys.index(key)] = 1
                y_label.append(label)
        x_train = np.array(x_train)
        y_label = np.array(y_label)
        scaler = MinMaxScaler(feature_range=(0,1))
        x_train = scaler.fit_transform(x_train)
        assert len(x_train) == len(y_label), f"x_train and y_label have different lengths!!!, {len(x_train)}, {len(y_label)}"
        return x_train, y_label

    # power consumption // plaintext input -----> Key Byte Value
    def power_input___key_id(traces_dict, max_size_test = 0):
        y_label = []
        x_train = []
        sorted_keys = sorted(traces_dict.keys())
        if max_size_test == 0:
            max_size_test = len(traces_dict[sorted_keys[0]]["power"])
        for key in sorted_keys:
            for i in range(max_size_test):
                x_train_iter = np.zeros(2)
                x_train_iter[0] = traces_dict[key]["power"][i]
                x_train_iter[1] = traces_dict[key]["plaintexts"][i]
                x_train.append(x_train_iter)
                label = np.zeros(len(sorted_keys))
                label[sorted_keys.index(key)] = 1
                y_label.append(label)
        x_train = np.array(x_train)
        y_label = np.array(y_label)

        scaler = MinMaxScaler(feature_range=(0,1))
        x_train = scaler.fit_transform(x_train)
        assert len(x_train) == len(y_label), f"x_train and y_label have different lengths!!!, {len(x_train)}, {len(y_label)}"
        return x_train, y_label

    # power consumption // plaintext input // S-Box HW -----> Key Byte Value
    def power_input_HW___key_id(traces_dict, max_size_test = 0):
        y_label = []
        x_train = []
        sorted_keys = sorted(traces_dict.keys())
        if max_size_test == 0:
            max_size_test = len(traces_dict[sorted_keys[0]]["power"])
        for key in sorted_keys:
            for i in range(max_size_test):
                x_train_iter = np.zeros(3)
                x_train_iter[0] = traces_dict[key]["power"][i]
                x_train_iter[1] = traces_dict[key]["plaintexts"][i]
                x_train_iter[2] = Auxiliar.hw[Auxiliar.sbox[(traces_dict[key]["plaintexts"][i] ^ key)]]
                x_train.append(x_train_iter)
                label = np.zeros(len(sorted_keys))
                label[sorted_keys.index(key)] = 1
                y_label.append(label)
        #x_train, y_label = shuffle(x_train, y_label)
        x_train = np.array(x_train)
        y_label = np.array(y_label)
        scaler = MinMaxScaler(feature_range=(0,1))
        x_train = scaler.fit_transform(x_train)
        assert len(x_train) == len(y_label), f"x_train and y_label have different lengths!!!, {len(x_train)}, {len(y_label)}"
        return x_train, y_label

    # power consumption // plaintext input // S-Box HW -----> Key Byte HW
    def power_input_HW___key_HW(traces_dict, max_size_test = 0):
        y_label = []
        x_train = []
        sorted_keys = sorted(traces_dict.keys())
        if max_size_test == 0:
            max_size_test = len(traces_dict[sorted_keys[0]]["power"])
        for key in sorted_keys:
            for i in range(max_size_test):
                x_train_iter = np.zeros(3)
                x_train_iter[0] = traces_dict[key]["power"][i]
                x_train_iter[1] = traces_dict[key]["plaintexts"][i]
                x_train_iter[2] = Auxiliar.hw[Auxiliar.sbox[(traces_dict[key]["plaintexts"][i] ^ key)]]
                x_train.append(x_train_iter)
                label = np.zeros(9)
                label[Auxiliar.hw[key]] = 1
                y_label.append(label)
        x_train = np.array(x_train)
        y_label = np.array(y_label)
        scaler = MinMaxScaler(feature_range=(0,1))
        x_train = scaler.fit_transform(x_train)
        assert len(x_train) == len(y_label), f"x_train and y_label have different lengths!!!, {len(x_train)}, {len(y_label)}"
        return x_train, y_label

    # power consumption -----> Key Byte Value
    def power___key_id(traces_dict, max_size_test = 0):
        y_label = []
        x_train = []
        sorted_keys = sorted(traces_dict.keys())
        if max_size_test == 0:
            max_size_test = len(traces_dict[sorted_keys[0]]["power"])
        for key in sorted_keys:
            for i in range(max_size_test):
                x_train.append(traces_dict[key]["plaintexts"][i])
                label = np.zeros(len(sorted_keys))
                label[sorted_keys.index(key)] = 1
                y_label.append(label)
        x_train = np.array(x_train)
        y_label = np.array(y_label)

        scaler = MinMaxScaler(feature_range=(0,1))
        x_train = scaler.fit_transform(x_train.reshape(-1, 1))
        assert len(x_train) == len(y_label), f"x_train and y_label have different lengths!!!, {len(x_train)}, {len(y_label)}"
        return x_train, y_label


# Class that contains the evaluation metrics for the model's success (Guessing Entropy)
class Eval_Metrics():
    
    # Method that plots the evolution of the correct key's GE while increasing the number of attack traces
    def plot_guessing_entropy_evolution(key_position_list, feature_options, model, n_epochs, callback, batch_size, layer_size):
        total_traces = len(key_position_list)
        g_e_evo = []
        for trace_amount in range(total_traces):
            g_e_evo.append(np.mean(key_position_list[:trace_amount+1]))
        
        # Plotting
        plt.figure(figsize=(18, 9))
        plt.plot(range(1, total_traces + 1), g_e_evo, linestyle='-', color='b', label='Guessing Entropy Evolution')
        plt.title('Guessing Entropy Evolution')
        plt.xlabel('Number of Traces')
        plt.ylabel('Guessing Entropy (GE)')
        plt.grid(True)
        plt.legend()
        plt.savefig(figures_path + f'GE_evo_{model.name}_{feature_options}_epochs{n_epochs}_callback{callback}_batch_size{batch_size}_layer_size{layer_size}_{total_traces}.png', format='png', dpi=300)  # You can change the format to 'pdf', 'svg', etc.
        return
    
    # Calculates the correct HW's GE and Rank
    def guessing_entropy_HW(traces_dict, model, number_of_keys, correct_key, x_test, y_test, n_epochs, batch_size, history, model_summary, feature_options, callback, layer_size):
        predictions = model.predict(x_test, verbose="auto")
        hw_position_list = [[] for _ in range(9)]
        correct_key_hw = y_test[0].argmax()
        for j in range(len(predictions)):
            arg_sort_reversed =list(reversed(predictions[j].argsort())) #highest probability HW in [0] and lowest in [8]
            for hw in arg_sort_reversed:
                hw_position_list[hw].append(arg_sort_reversed.index(hw))
        hw_GE_tuple_list = [(hw, np.mean(hw_position_list[hw])) for hw in range(9)]
        GE_tuple_list_sorted = sorted(hw_GE_tuple_list, key=lambda x: x[1])
        hw_rank_counter = 0
        hws_list = []
        for hw, ge in GE_tuple_list_sorted:
            if hw == correct_key_hw:
                hw_rank = hw_rank_counter
                hws_list.append(f'### CORRECT HW: {correct_key_hw}, GE: {ge}')
                print(f'### CORRECT HW: {correct_key_hw}, GE: {ge}')
            else:
                hws_list.append(f'HW: {hw}, GE: {ge}')
                print(f'HW: {hw}, GE: {ge}')
            hw_rank_counter += 1
        print("Average Index of correct Key HW in predictions:", np.mean(hw_position_list[correct_key_hw]), "HW Rank:", hw_rank, "Accuracy:", history.history['accuracy'][-1])
        Eval_Metrics.plot_guessing_entropy_evolution(hw_position_list[correct_key_hw], feature_options, model, n_epochs, callback, batch_size, layer_size)
        Eval_Metrics.__write_statistics_to_file(traces_dict, number_of_keys, hw_position_list[correct_key_hw], n_epochs, batch_size, history, False, model_summary, hws_list, hw_rank, feature_options, model, callback, layer_size)
        return np.mean(hw_position_list[correct_key_hw]), hw_rank

    # Calculates the correct Key's GE and Rank
    def guessing_entropy(traces_dict, model, number_of_keys, correct_key, x_test, y_test, n_epochs, batch_size, history, model_summary, feature_options, callback, layer_size):
        predictions = model.predict(x_test, verbose="auto")
        correct_key_position_list = []
        guessing_entropy_list = [[] for _ in range(0, number_of_keys)]
        sorted_keys = sorted(traces_dict.keys())
        correct_key_zero_indexed = sorted_keys.index(correct_key)

        for j in range(len(predictions)):
            arg_sort_reversed = list(reversed(predictions[j].argsort())) # most probable key in [0] and lowest in [...]
            #print()
            for key in arg_sort_reversed:
                if key == correct_key_zero_indexed:
                    correct_key_position_list.append(arg_sort_reversed.index(correct_key_zero_indexed))
                guessing_entropy_list[key].append(arg_sort_reversed.index(key))
        key_GE_tuple_list = [(i, np.mean(guessing_entropy_list[i])) for i in range(number_of_keys)] 
        key_GE_tuple_list_sorted = sorted(key_GE_tuple_list, key=lambda x: x[1])
        key_rank_counter = 0
        keys_list = []
        for key, ge in key_GE_tuple_list_sorted:
            if key == correct_key_zero_indexed:
                key_rank = key_rank_counter
                keys_list.append(f'### CORRECT Key: {sorted_keys[key]}, GE: {ge}')
                print(f'### CORRECT Key: {sorted_keys[key]}, GE: {ge}')
            else:
                keys_list.append(f'Key: {sorted_keys[key]}, GE: {ge}')
                print(f'Key: {sorted_keys[key]}, GE: {ge}')
            key_rank_counter += 1
        
        print(f'Average position of correct key {correct_key} (GE) = {str(np.mean(correct_key_position_list))} / {number_of_keys}. Key Rank: {key_rank}')
        Eval_Metrics.plot_guessing_entropy_evolution(correct_key_position_list, feature_options, model, n_epochs, callback, batch_size, layer_size)
        Eval_Metrics.__write_statistics_to_file(traces_dict, number_of_keys, correct_key_position_list, n_epochs, batch_size, history, False, model_summary, keys_list, key_rank, feature_options, model, callback, layer_size)
        return np.mean(correct_key_position_list), key_rank

    # Writes statistics to a given file
    def __write_statistics_to_file(traces_dict, number_of_keys, key_position_list, n_epochs, batch_size, history, conv, model_summary, keys_list, key_rank, feature_options, model, callback, layer_size):
        f = open(params_file, "a")
        f.write(f"{feature_options}\n")
        for key_str in keys_list:
            f.write(key_str + '\n')
        f.write(f'Amount of Traces = {len(traces_dict[0]["power"])}; # epochs = {n_epochs}; batch_size = {batch_size}; callback = {callback}; layer_size = {layer_size}; GE = {str(np.mean(key_position_list))} / {number_of_keys}; Key Rank = {key_rank}\n')
        f.write(model_summary)
        f.write(f"\nLoss: {history.history['loss'][-1]}, Accuracy: {history.history['accuracy'][-1]}\n")
        f.write('-------------------------------------------------------------\n\n\n\n\n')
        f.close()
        return


# Class that conducts the DL-SCA attacks, testing several parameter values
class DLSCA():

    # Constructor
    def __init__(self):
        self.traces_dict = Auxiliar.loadNPY_avg(traces_path)
        self.attack_traces_dict = Auxiliar.loadNPY_avg(attack_traces_path)
        self.correct_key = list(self.attack_traces_dict.keys())[0]
        self.classes = self.number_of_keys = len(self.traces_dict.keys())
    
    # Method that trains and tests all framework configurations
    def train_and_test(self, shorten_tests_int, shorten, test_index_start, retest):
        print("Power traces reading finished!!\n\n")
        prompt_str = f"""- Energy (J), Δclock_cycles, plaintext_input -> key_byte (0)\n- Energy (J), Δclock_cycles, plaintext_input, sbox_HW -> key_byte (1)\n- Power (W), plaintext_input -> key_byte (2)\n- Power (W), plaintext_input, sbox_HW -> key_byte (3)\n- Power (W), plaintext_input, sbox_HW -> key_byte_HW (4)\n- Power (W) -> key_byte (5)"""
        print(f"Testing the following input feature options\n{prompt_str}\n")
            
        
        # HyperParameter Options
        layer_sizes = [10, 100]
        epochs = [100, 200]
        batch_sizes = [32, 512]
        callbacks_list = [[], [Auxiliar.reduce_lr]]
        #####

        if shorten:
            layer_sizes = [10]
            epochs = [50]
            batch_sizes = [32]
            callbacks_list = [[Auxiliar.reduce_lr]]

        input_feature_options = [
            #Diff_Input_Features.energy_cycles_input___key_id,
            Diff_Input_Features.energy_cycles_input_HW___key_id,
            Diff_Input_Features.power_input___key_id,
            Diff_Input_Features.power_input_HW___key_id,
            Diff_Input_Features.power_input_HW___key_HW,
            Diff_Input_Features.power___key_id
        ]
        models = [
            Models.aisy_cnn,
            Models.aisy_mlp
        ]
        
        # Estimate Time Complexity
        iter_dict = {"inputs": len(input_feature_options),
            "callbacks": len(callbacks_list),
            "layer_sizes": len(layer_sizes),
            "epochs": len(epochs),
            "batch_size": len(batch_sizes),
            "models": len(models)}
        total_factorial = 1
        for key in iter_dict.keys():
            total_factorial = total_factorial * iter_dict[key]
        print("Total amount of combinations:", total_factorial)
        print("Total Hours:", total_factorial * 2)
        print("Total days:", (total_factorial * 2) / 24)
        
        # Create List with all parameter configurations
        parameters_dict_list = []
        for layer_size in layer_sizes:
            for model_epochs in epochs:
                for model_batch_size in batch_sizes:
                    for callback in callbacks_list:
                        for input_feature_option in input_feature_options:
                            for model in models:
                                parameters_dict_list.append({
                                        "layer_size": layer_size,
                                        "model_epochs": model_epochs,
                                        "model_batch_size": model_batch_size,
                                        "callback": callback,
                                        "input_feature_option": input_feature_option,
                                        "model": model
                                    })
        parameters_dict_list_original_length = len(parameters_dict_list)
        # If tests were interrupted, they will resume at test_index_start variable
        parameters_dict_list = parameters_dict_list[test_index_start:]
        print("Length of parameters list:", len(parameters_dict_list))
        parameter_iter = parameters_dict_list_original_length - len(parameters_dict_list)
        
        # Calculate number of impossible configurations for the chosen DL frameworks (CNN w/ only 1 feature)
        skipped_counter=0
        for parameters_dict in parameters_dict_list:
            layer_size, model_epochs, model_batch_size, callback, input_feature_option, model = Auxiliar.decapsulate_parameters_dict(parameters_dict)
            if "cnn" in model.__name__ and "power___key_id" in input_feature_option.__name__:
                skipped_counter += 1
                print("CNN for only 1 feature skipped!!!!\n\n")
        print("Number of iters:", len(parameters_dict_list) - skipped_counter, len(parameters_dict_list), skipped_counter)
        
        ##OVERWRITE RETEST - retest the most successful configurations
        if retest:
            parameters_dict_list = []
            for i in range(4):
                parameters_dict_list.append({
                                            "layer_size": 100,
                                            "model_epochs": 100,
                                            "model_batch_size": 512,
                                            "callback": [],
                                            "input_feature_option": Diff_Input_Features.power_input___key_id,
                                            "model": Models.aisy_cnn
                                            })
                parameters_dict_list.append({
                                            "layer_size": 100,
                                            "model_epochs": 100,
                                            "model_batch_size": 512,
                                            "callback": [],
                                            "input_feature_option": Diff_Input_Features.energy_cycles_input_HW___key_id,
                                            "model": Models.aisy_mlp
                                            })
                parameters_dict_list.append({
                                            "layer_size": 100,
                                            "model_epochs": 100,
                                            "model_batch_size": 512,
                                            "callback": [],
                                            "input_feature_option": Diff_Input_Features.power___key_id,
                                            "model": Models.aisy_mlp
                                            })
            parameters_dict_list = parameters_dict_list[5:]
            print("SIZE OF RETEST:", len(parameters_dict_list))


        
        with open('test_results.json', 'a') as f:
            f.write("[")
        #shorten_tests_int = 50 #if 0 then full tests, else this sets the amount of traces per key
        for parameters_dict in parameters_dict_list:
            parameter_iter += 1
            print(f"On index {parameter_iter}/{parameters_dict_list_original_length}\n")
            layer_size, model_epochs, model_batch_size, callback, input_feature_option, model = Auxiliar.decapsulate_parameters_dict(parameters_dict)
            #Input feature option only has one feature so it is impossible to work with cnn. Therefore we skip this iteration
            if "cnn" in model.__name__ and "power___key_id" in input_feature_option.__name__:
                print("CNN for only 1 feature skipped!!!!\n\n")
                continue
            print("model", model.__name__, "input_method", input_feature_option.__name__)
            # Training dataset
            x_train, y_label = input_feature_option(self.traces_dict, shorten_tests_int)
            # Testing dataset
            x_test, y_test = input_feature_option(self.attack_traces_dict, shorten_tests_int)
            number_of_features = len(x_train[0])
            if "cnn" in model.__name__:
                x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
                x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
            print("SHAPE OF X_TRAIN", x_train.shape, "NUMBER OF FEATURES", number_of_features)
            print("Length of training traces:", len(x_train), "for", len(self.traces_dict.keys()), "different keys")
            print("layer_size", layer_size,
                    "epochs", model_epochs, 
                    "batch_size", model_batch_size, 
                    "callback", callback)
            if "___key_HW" in input_feature_option.__name__:
                #Model made to check HW, initialize mlp with different number of classes
                self.model, self.model_summary_str = model(9, number_of_features, layer_size)
            else:
                # Regular Key ID classification
                self.model, self.model_summary_str = model(self.classes, number_of_features, layer_size)
            print("Fitting Model...\n\n")
            history = self.model.fit(x = x_train, y = y_label, batch_size = model_batch_size, epochs = model_epochs, validation_split=0.1, callbacks = callback, verbose=0, shuffle=True)
            print("Fitting Done! Testing attack traces...\n\n")
            # Classify GE for the current framework configuration
            if "___key_HW" in input_feature_option.__name__:
                ge, key_rank = Eval_Metrics.guessing_entropy_HW(traces_dict=self.traces_dict, 
                                                                model=self.model, 
                                                                number_of_keys=self.number_of_keys, 
                                                                correct_key=self.correct_key, 
                                                                x_test=x_test, 
                                                                y_test=y_test, 
                                                                n_epochs = model_epochs, 
                                                                batch_size = model_batch_size, 
                                                                history=history, 
                                                                model_summary=self.model_summary_str, 
                                                                feature_options = input_feature_option.__name__,
                                                                callback = str(callback),
                                                                layer_size = layer_size)
            else:
                ge, key_rank = Eval_Metrics.guessing_entropy(traces_dict=self.traces_dict, 
                                                                model=self.model, 
                                                                number_of_keys=self.number_of_keys, 
                                                                correct_key=self.correct_key, 
                                                                x_test=x_test, 
                                                                y_test=y_test, 
                                                                n_epochs = model_epochs, 
                                                                batch_size = model_batch_size, 
                                                                history=history, 
                                                                model_summary=self.model_summary_str, 
                                                                feature_options = input_feature_option.__name__,
                                                                callback = str(callback),
                                                                layer_size = layer_size)
            test_results = {"layer_size": layer_size, 
                            "model_epochs": model_epochs, 
                            "model_batch_size": model_batch_size, 
                            "callback": str(callback), 
                            "GE": ge, 
                            "key_rank": key_rank, 
                            "feature_option": input_feature_option.__name__, 
                            "accuracy": history.history['accuracy'][-1], 
                            "model": model.__name__ }
            # appends test results plus a comma 
            with open('test_results.json', 'a') as f:
                json.dump(test_results, f, indent=4)
                f.write(",")
            finished_iter_time = datetime.now()
            print("finished_iter_time", str(finished_iter_time).split('.')[0])
            
        #removes last comma
        with open('test_results.json', 'r') as f:
            file_content = f.read()
        with open('test_results.json', 'w') as f:
            f.write(file_content[:-1])
        with open('test_results.json', 'a') as f:
            f.write(']')
        

        # Open results file and sort by ge 
        #with open('test_results.json') as f:
        #    my_dict = json.load(f)
        #    print("my_dict", my_dict)
        #    sorted_dict = sorted(my_dict, key=lambda x: x["GE"], reverse=True)
        #    for line in sorted_dict:
        #        print(line)


# UNUSED - Classical Template Attack
class TemplateAttack():

    # default constructor
    def __init__(self, number_of_keys, correct_key, numPOIs, POIspacing):
        self.tracesDict = Auxiliar.loadNPY(self, traces_path)
        self.attack_traces = Auxiliar.loadNPY(self, attack_traces_path)
        #self.plaintexts_list = Auxiliar.readPlaintexts(plaintexts_path)
        #self.attack_plaintexts_list = Auxiliar.readPlaintexts(attack_plaintexts_path)
        self.number_of_keys = number_of_keys
        self.correct_key = correct_key
        self.trace_length = len(self.tracesDict[0]["power"])
        self.numPOIs = numPOIs
        self.POIspacing = POIspacing
        #print("Number of Keys:", len(self.tracesDict))
        #print("Number of sets:", len(self.tracesDict[0]))
        #print("Number of values per set:", len(self.tracesDict[0][0]))

    # splits traces by HW for different plaintexts
    def __split_traces_by_HW(self):
        keys_list = list(self.tracesDict.keys())
        #print("keys_list", keys_list)
        tempTracesHW = [[] for _ in range(9)]
        for key in keys_list:
            tempSbox = [Auxiliar.sbox[self.tracesDict[key]["plaintexts"][i] ^ key] for i in range(len(self.tracesDict[key]["plaintexts"]))] 
            tempHW   = [Auxiliar.hw[s] for s in tempSbox]
            
            for i in range(len(self.tracesDict[key]["power"])):
                HW = tempHW[i]
                tempTracesHW[HW].append(self.tracesDict[key]["power"][i])

        return [np.array(tempTracesHW[HW]) for HW in range(9)]
    
    #calculate POIs list
    def __find_POIs(self, tempTracesHW, numPOIs, POIspacing):
        tempMeans = np.zeros((9, self.trace_length))
    
        for i in range(9):
            tempMeans[i] = np.average(tempTracesHW[i], 0)
            tempSumDiff = np.zeros(self.trace_length)
            for i in range(9):
                for j in range(i):
                    tempSumDiff += np.abs(tempMeans[i] - tempMeans[j])
        # 5: Find POIs
        POIs = []
        for i in range(numPOIs):
            # Find the max
            nextPOI = tempSumDiff.argmax()
            POIs.append(nextPOI)
            
            # Make sure we don't pick a nearby value
            poiMin = max(0, nextPOI - POIspacing)
            poiMax = min(nextPOI + POIspacing, len(tempSumDiff))
            for j in range(poiMin, poiMax):
                tempSumDiff[j] = 0

        return tempMeans, POIs

    #create templates for each HW
    def __fill_Matrixes(self, tempMeans, POIs, numPOIs, tempTracesHW):
        # 6: Fill up mean and covariance matrix for each HW
        meanMatrix = np.zeros((9, numPOIs))
        covMatrix  = np.zeros((9, numPOIs, numPOIs))
        for HW in range(9):
            for i in range(numPOIs):
                # Fill in mean
                meanMatrix[HW][i] = tempMeans[HW][POIs[i]]
                for j in range(numPOIs):
                    x = tempTracesHW[HW][POIs[i]]
                    y = tempTracesHW[HW][POIs[j]]
                    covMatrix[HW,i,j] = Auxiliar.cov(x, y)
        return POIs, meanMatrix, covMatrix
    
    def profiling_phase(self):

        tempTracesHW = self.__split_traces_by_HW()

        tempMeans, POIs = self.__find_POIs(tempTracesHW, self.numPOIs, self.POIspacing)

        return self.__fill_Matrixes(tempMeans, POIs, self.numPOIs, tempTracesHW)
    
    def attack_phase(self, POIs, meanMatrix, covMatrix):
        self.attack_traces = self.attack_traces[self.correct_key]
        print("Length of attack traces:", len(self.attack_traces))
        #print("Length of each attack traces:", len(self.attack_traces[0]))
        key_position_list = []
        guessing_entropy_list = [[] for _ in range(0, self.number_of_keys)]
        #print("Attack traces length:", len(atkTraces))
        P_k = np.zeros(self.number_of_keys)
        for j in range(len(self.attack_traces)):
            # Grab key points and put them in a matrix
            a = [self.attack_traces["power"][j]]
            
            # Test each key
            for k in range(0, self.number_of_keys):
                # Find HW coming out of sbox
                HW = Auxiliar.hw[Auxiliar.sbox[self.attack_traces["plaintexts"][j] ^ k]]
            
                # Find p_{k,j}
                rv = multivariate_normal(meanMatrix[HW], covMatrix[HW])
                # print(meanMatrix[HW])
                p_kj = rv.pdf(a)
        
                # Add it to running total
                P_k[k] += np.log(p_kj)
                # P_k[k] += p_kj

            # Print our top 5 results so far
            # Best match on the right
            print("Attack Trace ", j, P_k.argsort()[-10:])
            for pos in P_k.argsort():
                if pos == self.correct_key:
                    #print("Pos: ", list(P_k.argsort()).index(25))
                    key_position_list.append(self.number_of_keys - 1 - list(P_k.argsort()).index(self.correct_key))
            for pos in P_k.argsort():
                guessing_entropy_list[pos].append(self.number_of_keys - 1 - list(P_k.argsort()).index(pos))

        key_GE_tuple_list = [(i, np.mean(guessing_entropy_list[i])) for i in range(self.number_of_keys)] 
        key_GE_tuple_list_sorted = sorted(key_GE_tuple_list, key=lambda x: x[1]) 
        for key, ge in key_GE_tuple_list_sorted:
            if key == self.correct_key:
                print(f'### CORRECT Key: {key}, GE: {ge}')
            else:
                print(f'Key: {key}, GE: {ge}')
        
        print(f'Average position of correct key {self.correct_key} (GE) = {str(np.mean(key_position_list))} / {self.number_of_keys}')
        # write history to file
        f = open("params_history.txt", "a")
        f.write(f'Traces size = {len(self.tracesDict[0][0])}; # POIS = {self.numPOIs}; POIspacing = {self.POIspacing}; GE = {str(np.mean(key_position_list))} / {self.number_of_keys}\n')
        f.close()


# Class that conducts a simplified Template Attack
class TemplateAttackPDF():
    # default constructor
    def __init__(self, number_of_keys, correct_key):
        self.tracesDict = Auxiliar.loadNPY_avg(traces_path)
        self.attack_traces = Auxiliar.loadNPY_avg(attack_traces_path)
        self.number_of_keys = number_of_keys
        self.correct_key = correct_key

    # Method that conducts the attack by creating PDFs for each key guess, and compares them by multiplication of log probability
    def attack(self):
        PDFs = {}
        #key_position_list = []
        guessing_entropy_list = [[] for _ in range(0, self.number_of_keys)]
        sorted_keys = sorted(self.tracesDict.keys())
        correct_key_zero_indexed = sorted_keys.index(self.correct_key)
        #probability_storage = np.zeros(len(sorted_keys))
        probability_storage = np.array([1 for _ in range(len(sorted_keys))])
        for key_index in range(len(sorted_keys)):
            PDFs[sorted_keys[key_index]] = stats.norm(np.mean(self.tracesDict[sorted_keys[key_index]]["power"]), np.std(self.tracesDict[sorted_keys[key_index]]["power"]))
        
        for j in range(len(self.attack_traces[self.correct_key]["power"])):
            for key_index in range(len(sorted_keys)):
                probability_storage[key_index] *= np.log(PDFs[sorted_keys[key_index]].pdf(self.attack_traces[self.correct_key]["power"][j]))
            argsort_reversed = list(reversed(probability_storage.argsort())) # 0 index equals highest probability
            print("Attack Trace ", j, argsort_reversed[-10:])
            for pos in argsort_reversed:
                guessing_entropy_list[pos].append(list(argsort_reversed).index(pos))
        key_GE_tuple_list = [(i, np.mean(guessing_entropy_list[i])) for i in range(self.number_of_keys)] 
        key_GE_tuple_list_sorted = sorted(key_GE_tuple_list, key=lambda x: x[1])

        key_rank_counter = 0
        for key_index, ge in key_GE_tuple_list_sorted:
            if sorted_keys[key_index] == self.correct_key:
                print(f'### CORRECT Key: {sorted_keys[key_index]}, GE: {ge}')
                key_rank = key_rank_counter
            else:
                print(f'Key: {sorted_keys[key_index]}, GE: {ge}')
            key_rank_counter += 1
        x_axis = np.arange(len(self.attack_traces[self.correct_key]["power"]))
        ge_list_evo_correct_key = []
        print("KEY RANK: ", key_rank)
        for trace_amount in range(len(self.attack_traces[self.correct_key]["power"])):
            ge_list_evo_correct_key.append(np.mean(guessing_entropy_list[correct_key_zero_indexed][:trace_amount+1]))
            
        plt.figure(figsize=(18, 9))
        plt.plot(x_axis, ge_list_evo_correct_key,label=f"Key {self.correct_key} GE Over Time", color="b")
        plt.xlabel("Attack Trace Index")
        plt.ylabel("Guessing Entropy")
        plt.title("Guessing Entropy Over Time for Template Attack")
        plt.legend()
        plt.grid(True)
        dt = datetime.now()
        ts = datetime.timestamp(dt)
        plt.savefig(figures_path + f'template_attack_correct_key_evo_{ts}.png')


def main():
    start = datetime.now()
    print("Start time", str(start).split('.')[0])
    dlsca = DLSCA()
    dlsca.train_and_test(shorten_tests_int= 0, shorten= False, test_index_start=0, retest=True)

    #Auxiliar.analyze_hws()

    # TEMPLATE ATTACK w/o HW
    #ta = TemplateAttack(number_of_keys=4, correct_key=50, numPOIs=1, POIspacing=3)
    #POIs, meanMatrix, covMatrix =  ta.profiling_phase()
    #ta.attack_phase(POIs, meanMatrix, covMatrix)
    
    #ta = TemplateAttackPDF(number_of_keys=64, correct_key=16)
    #ta.attack()

    end = datetime.now()
    time_taken = end - start

    # Remove the decimal part of seconds
    formatted_time = str(time_taken).split('.')[0]

    print("Time to run", formatted_time)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


if __name__ == "__main__":
    main()