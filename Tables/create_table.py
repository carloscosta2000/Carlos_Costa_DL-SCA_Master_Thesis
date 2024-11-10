import json


def main():
#\textbf{Model} & \textbf{Input Feature Option} & \textbf{Layer Size} & \textbf{Epochs} & \textbf{Batch Size} & \textbf{Callback} & \textbf{GE} & \textbf{KR}
    with open('test_results.json') as f:
        my_dict_list = json.load(f)
        sorted_dict_list = sorted(my_dict_list, key=lambda x: (x['model'], x['feature_option'], x['layer_size'], x['model_epochs'], x['model_batch_size'], x['callback']))
    create_table_text_from_dict(sorted_dict_list, "table_lines.txt")
    

# Creates a Latex table from the input dictionary
def create_table_text_from_dict(my_dict_list, filename):
    table_lines = []
    for dict in my_dict_list:
        separator = "\t & "
        if dict["callback"] == "[]":
            callback_str = "_"
        else:
            callback_str = "RLROP"
        text_line = dict["model"] + separator + dict["feature_option"] + separator + str(dict["layer_size"]) + separator + str(dict["model_epochs"]) + separator + str(dict["model_batch_size"]) + separator + callback_str + separator + str(dict["GE"]) + separator + str(dict["key_rank"]) + "\\\ \hline \n"
        text_line = text_line.replace("_", "\\_")
        table_lines.append(text_line)
    with open(filename, 'w') as file:
        file.writelines(table_lines)
if __name__ == "__main__":
    main()