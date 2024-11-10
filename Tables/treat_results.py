import json

# Open results file and sort by ge 
with open('test_results_joined.json') as f:
    my_dict_list = json.load(f)
    print("my_dict len", len(my_dict_list))
    #print("my_dict", my_dict_list)
    
    #sorted_dict_list = sorted(my_dict_list, key=lambda x: x["key_rank"], reverse=True)
    #for line in sorted_dict_list:
    #    print(line, "\n")

    # Define the keys you want to compare
    keys_to_compare = ['layer_size', 'model_epochs', 'model_batch_size', 'callback', 'feature_option', 'model']

    def find_repeated_values(dict_list, keys):
        seen_entries = {}
        repeated_entries = []

        # Iterate through the list of dictionaries
        for d in dict_list:
            # Extract the values for the specified keys
            current_values = tuple(d[key] for key in keys)
            
            # If we've seen this combination before, it's a duplicate
            if current_values in seen_entries:
                repeated_entries.append(current_values)
            else:
                # Mark this combination as seen
                seen_entries[current_values] = d
        
        return repeated_entries

    # Find repeated entries where all key-value pairs for the specified keys, including 'model', are the same
    #repeated_entries = find_repeated_values(my_dict_list, keys_to_compare)

    # Output the result
    #if repeated_entries:
        #print(f"Repeated key-value combinations found for keys {keys_to_compare}: {repeated_entries}")
    #else:
        #print(f"No repeated key-value combinations for keys: {keys_to_compare}")

    def my_find_repeated_values(dict_list, keys):
        seen_dict_list = []
        seen_dict_list.append(dict_list[0])
        repeated_values = []
        dict_list_no_fst = dict_list[1:]
        for dict in dict_list_no_fst:
            found_match = False
            for seen_dict in seen_dict_list:
                #print(dict["model"] == seen_dict["model"], dict["model"], seen_dict["model"])
                if (dict["layer_size"] == seen_dict["layer_size"] and 
                    dict["model_epochs"] == seen_dict["model_epochs"] and 
                    dict["model_batch_size"] == seen_dict["model_batch_size"] and 
                    dict["callback"] == seen_dict["callback"] and 
                    dict["feature_option"] == seen_dict["feature_option"] and 
                    dict["model"] == seen_dict["model"]):
                    found_match = True
                    repeated_values.append(dict)
                    break
        
            # If no match found, add the dictionary to seen_dict_list
            if not found_match:
                seen_dict_list.append(dict)
        return repeated_values
    repeated_entries = my_find_repeated_values(my_dict_list, keys_to_compare)
    print(repeated_entries)