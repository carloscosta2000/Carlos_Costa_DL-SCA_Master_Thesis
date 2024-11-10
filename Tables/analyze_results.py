import json
import numpy as np
import create_table


def main():
    with open('test_results.json') as f:
        my_dict_list = json.load(f)
        possible_values = {"model": [], "feature_option":[], "layer_size":[], "model_epochs":[], "model_batch_size": [], "callback": []}
        for dict in my_dict_list:
            for key in possible_values.keys():
                if dict[key] not in possible_values[key]:
                    possible_values[key].append(dict[key])
    analyze = False
    if analyze:
        print(possible_values)
        analyze_models(my_dict_list)
        analyze_input_feature_options(my_dict_list)
        analyze_layer_size(my_dict_list)
        analyze_epochs(my_dict_list)
        analyze_batch_size(my_dict_list)
        analyze_callbacks(my_dict_list)
    analyze_input_feature_options_with_models(my_dict_list)    
    #top_n_results_no_hw_target(my_dict_list, 10)

# Prints average performance of each Model
def analyze_models(my_dict_list):
    #model
    model_mlp = {"GE": [], "KR": []}
    model_cnn = {"GE": [], "KR": []}
    for dict in my_dict_list:
        if dict["model"] == "aisy_cnn":
            model_cnn["GE"].append(dict["GE"])
            model_cnn["KR"].append(dict["key_rank"])
        elif dict["model"] == "aisy_mlp":
            model_mlp["GE"].append(dict["GE"])
            model_mlp["KR"].append(dict["key_rank"])
    print(f"CNN: AVG GE: {np.average(model_cnn['GE'])}, AVG KR: {np.average(model_cnn['KR'])}")
    print(f"MLP: AVG GE: {np.average(model_mlp['GE'])}, AVG KR: {np.average(model_mlp['KR'])}\n\n\n")

# Prints average performance of each Input Feature Option
def analyze_input_feature_options(my_dict_list):
    #input feature option
    energy_cycles_input_HW___key_id = {"GE": [], "KR": []}
    power_input___key_id = {"GE": [], "KR": []}
    power_input_HW___key_id = {"GE": [], "KR": []}
    power_input_HW___key_HW = {"GE": [], "KR": []}
    power___key_id = {"GE": [], "KR": []}
    for dict in my_dict_list:
        if dict["feature_option"] == "energy_cycles_input_HW___key_id":
            energy_cycles_input_HW___key_id["GE"].append(dict["GE"])
            energy_cycles_input_HW___key_id["KR"].append(dict["key_rank"])
        elif dict["feature_option"] == "power_input___key_id":
            power_input___key_id["GE"].append(dict["GE"])
            power_input___key_id["KR"].append(dict["key_rank"])
        elif dict["feature_option"] == "power_input_HW___key_id":
            power_input_HW___key_id["GE"].append(dict["GE"])
            power_input_HW___key_id["KR"].append(dict["key_rank"])
        elif dict["feature_option"] == "power_input_HW___key_HW":
            power_input_HW___key_HW["GE"].append(dict["GE"])
            power_input_HW___key_HW["KR"].append(dict["key_rank"])
        elif dict["feature_option"] == "power___key_id":
            power___key_id["GE"].append(dict["GE"])
            power___key_id["KR"].append(dict["key_rank"])
    print(f"energy_cycles_input_HW___key_id: AVG GE: {np.average(energy_cycles_input_HW___key_id['GE'])}, AVG KR: {np.average(energy_cycles_input_HW___key_id['KR'])}")
    print(f"power_input___key_id: AVG GE: {np.average(power_input___key_id['GE'])}, AVG KR: {np.average(power_input___key_id['KR'])}")
    print(f"power_input_HW___key_id: AVG GE: {np.average(power_input_HW___key_id['GE'])}, AVG KR: {np.average(power_input_HW___key_id['KR'])}")
    print(f"power_input_HW___key_HW: AVG GE: {np.average(power_input_HW___key_HW['GE'])}, AVG KR: {np.average(power_input_HW___key_HW['KR'])}")
    print(f"power___key_id: AVG GE: {np.average(power___key_id['GE'])}, AVG KR: {np.average(power___key_id['KR'])}\n\n\n")

# Prints average performance of each Model in combination with each Input Feature Option    
def analyze_input_feature_options_with_models(my_dict_list):
    #input feature option
    energy_cycles_input_HW___key_id_mlp = {"GE": [], "KR": []}
    power_input___key_id_mlp = {"GE": [], "KR": []}
    power_input_HW___key_id_mlp = {"GE": [], "KR": []}
    power_input_HW___key_HW_mlp = {"GE": [], "KR": []}
    power___key_id_mlp = {"GE": [], "KR": []}

    energy_cycles_input_HW___key_id_cnn = {"GE": [], "KR": []}
    power_input___key_id_cnn = {"GE": [], "KR": []}
    power_input_HW___key_id_cnn = {"GE": [], "KR": []}
    power_input_HW___key_HW_cnn = {"GE": [], "KR": []}
    power___key_id_cnn = {"GE": [], "KR": []}
    for dict in my_dict_list:
        if dict["feature_option"] == "energy_cycles_input_HW___key_id":
            if dict["model"] == "aisy_mlp":
                energy_cycles_input_HW___key_id_mlp["GE"].append(dict["GE"])
                energy_cycles_input_HW___key_id_mlp["KR"].append(dict["key_rank"])
            elif dict["model"] == "aisy_cnn":
                energy_cycles_input_HW___key_id_cnn["GE"].append(dict["GE"])
                energy_cycles_input_HW___key_id_cnn["KR"].append(dict["key_rank"])
        elif dict["feature_option"] == "power_input___key_id":
            if dict["model"] == "aisy_mlp":
                power_input___key_id_mlp["GE"].append(dict["GE"])
                power_input___key_id_mlp["KR"].append(dict["key_rank"])
            elif dict["model"] == "aisy_cnn":
                power_input___key_id_cnn["GE"].append(dict["GE"])
                power_input___key_id_cnn["KR"].append(dict["key_rank"])
        elif dict["feature_option"] == "power_input_HW___key_id":
            if dict["model"] == "aisy_mlp":
                power_input_HW___key_id_mlp["GE"].append(dict["GE"])
                power_input_HW___key_id_mlp["KR"].append(dict["key_rank"])
            elif dict["model"] == "aisy_cnn":
                power_input_HW___key_id_cnn["GE"].append(dict["GE"])
                power_input_HW___key_id_cnn["KR"].append(dict["key_rank"])
        elif dict["feature_option"] == "power_input_HW___key_HW":
            if dict["model"] == "aisy_mlp":
                power_input_HW___key_HW_mlp["GE"].append(dict["GE"])
                power_input_HW___key_HW_mlp["KR"].append(dict["key_rank"])
            elif dict["model"] == "aisy_cnn":
                power_input_HW___key_HW_cnn["GE"].append(dict["GE"])
                power_input_HW___key_HW_cnn["KR"].append(dict["key_rank"])
        elif dict["feature_option"] == "power___key_id":
            if dict["model"] == "aisy_mlp":
                power___key_id_mlp["GE"].append(dict["GE"])
                power___key_id_mlp["KR"].append(dict["key_rank"])
            elif dict["model"] == "aisy_cnn":
                power___key_id_cnn["GE"].append(dict["GE"])
                power___key_id_cnn["KR"].append(dict["key_rank"])
    print(f"energy_cycles_input_HW___key_id_mlp: AVG GE: {np.average(energy_cycles_input_HW___key_id_mlp['GE'])}, AVG KR: {np.average(energy_cycles_input_HW___key_id_mlp['KR'])}")
    print(f"power_input___key_id_mlp: AVG GE: {np.average(power_input___key_id_mlp['GE'])}, AVG KR: {np.average(power_input___key_id_mlp['KR'])}")
    print(f"power_input_HW___key_id_mlp: AVG GE: {np.average(power_input_HW___key_id_mlp['GE'])}, AVG KR: {np.average(power_input_HW___key_id_mlp['KR'])}")
    print(f"power_input_HW___key_HW_mlp: AVG GE: {np.average(power_input_HW___key_HW_mlp['GE'])}, AVG KR: {np.average(power_input_HW___key_HW_mlp['KR'])}")
    print(f"power___key_id_mlp: AVG GE: {np.average(power___key_id_mlp['GE'])}, AVG KR: {np.average(power___key_id_mlp['KR'])}\n\n\n")

    print(f"energy_cycles_input_HW___key_id_cnn: AVG GE: {np.average(energy_cycles_input_HW___key_id_cnn['GE'])}, AVG KR: {np.average(energy_cycles_input_HW___key_id_cnn['KR'])}")
    print(f"power_input___key_id_cnn: AVG GE: {np.average(power_input___key_id_cnn['GE'])}, AVG KR: {np.average(power_input___key_id_cnn['KR'])}")
    print(f"power_input_HW___key_id_cnn: AVG GE: {np.average(power_input_HW___key_id_cnn['GE'])}, AVG KR: {np.average(power_input_HW___key_id_cnn['KR'])}")
    print(f"power_input_HW___key_HW_cnn: AVG GE: {np.average(power_input_HW___key_HW_cnn['GE'])}, AVG KR: {np.average(power_input_HW___key_HW_cnn['KR'])}")
    print(f"power___key_id: AVG GE: {np.average(power___key_id_cnn['GE'])}, AVG KR: {np.average(power___key_id_cnn['KR'])}\n\n\n")

# Prints average performance of each Layer Size     
def analyze_layer_size(my_dict_list):
    #model
    ls_10 = {"GE": [], "KR": []}
    ls_100 = {"GE": [], "KR": []}
    for dict in my_dict_list:
        if dict["layer_size"] == 10:
            ls_10["GE"].append(dict["GE"])
            ls_10["KR"].append(dict["key_rank"])
        elif dict["layer_size"] == 100:
            ls_100["GE"].append(dict["GE"])
            ls_100["KR"].append(dict["key_rank"])
    print(f"ls_10: AVG GE: {np.average(ls_10['GE'])}, AVG KR: {np.average(ls_10['KR'])}")
    print(f"ls_100: AVG GE: {np.average(ls_100['GE'])}, AVG KR: {np.average(ls_100['KR'])}\n\n\n")

# Prints average performance of each Epoch amount
def analyze_epochs(my_dict_list):
    #model
    ep_100 = {"GE": [], "KR": []}
    ep_200 = {"GE": [], "KR": []}
    for dict in my_dict_list:
        if dict["model_epochs"] == 100:
            ep_100["GE"].append(dict["GE"])
            ep_100["KR"].append(dict["key_rank"])
        elif dict["model_epochs"] == 200:
            ep_200["GE"].append(dict["GE"])
            ep_200["KR"].append(dict["key_rank"])
    print(f"ep_100: AVG GE: {np.average(ep_100['GE'])}, AVG KR: {np.average(ep_100['KR'])}")
    print(f"ep_200: AVG GE: {np.average(ep_200['GE'])}, AVG KR: {np.average(ep_200['KR'])}\n\n\n")

# Prints average performance of each Batch Size number
def analyze_batch_size(my_dict_list):
    #model
    bs_32 = {"GE": [], "KR": []}
    bs_512 = {"GE": [], "KR": []}
    for dict in my_dict_list:
        if dict["model_batch_size"] == 32:
            bs_32["GE"].append(dict["GE"])
            bs_32["KR"].append(dict["key_rank"])
        elif dict["model_batch_size"] == 512:
            bs_512["GE"].append(dict["GE"])
            bs_512["KR"].append(dict["key_rank"])
    print(f"bs_32: AVG GE: {np.average(bs_32['GE'])}, AVG KR: {np.average(bs_32['KR'])}")
    print(f"bs_512: AVG GE: {np.average(bs_512['GE'])}, AVG KR: {np.average(bs_512['KR'])}\n\n\n")

# Prints average performance of each Callback
def analyze_callbacks(my_dict_list):
    #model
    no_cb = {"GE": [], "KR": []}
    rlrop = {"GE": [], "KR": []}
    for dict in my_dict_list:
        if dict["callback"] == '[]':
            no_cb["GE"].append(dict["GE"])
            no_cb["KR"].append(dict["key_rank"])
        elif dict["callback"] == '[<tensorflow.python.keras.callbacks.ReduceLROnPlateau object at 0x7f2185ad29a0>]':
            rlrop["GE"].append(dict["GE"])
            rlrop["KR"].append(dict["key_rank"])
    print(f"no_cb: AVG GE: {np.average(no_cb['GE'])}, AVG KR: {np.average(no_cb['KR'])}")
    print(f"rlrop: AVG GE: {np.average(rlrop['GE'])}, AVG KR: {np.average(rlrop['KR'])}\n\n\n")

# Creates a Latex table for the top N results in the total results scope
def top_n_results_no_hw_target(my_dict_list, n):
    filtered_dict_no_hws = []
    for dict in my_dict_list:
        if dict["feature_option"] == "power_input_HW___key_HW":
            continue
        else:
            filtered_dict_no_hws.append(dict)
    sorted_dict = sorted(filtered_dict_no_hws, key=lambda x: x['key_rank'])
    print(sorted_dict[:n])
    create_table.create_table_text_from_dict(sorted_dict[:n], "top_10.txt")
if __name__ == "__main__":
    main()