import dlsca as dlsca
import numpy as np

# HW - amount of keys for that HW
keys_amount_dictionary = {0: 1, #
                          1: 8,
                          2: 8,
                          3: 9,
                          4: 12,
                          5: 9,
                          6: 8,
                          7: 8,
                          8: 1}

def main():
    hws = [[] for _ in range(9)]
    for key in range(256):
        hws[dlsca.Auxiliar.hw[key]].append(key)
    for hw in range(9):
        hws[hw] = list(reversed(sorted(hws[hw])))[:keys_amount_dictionary[hw]]
        print(f"HW{hw}", hws[hw])
    flattened_key_list = sorted(dlsca.Auxiliar.my_flatten(hws))
    print(flattened_key_list)
    # Saves array of the keys to be used in the framework
    np.save("keys_array.npy", flattened_key_list)

if __name__ == "__main__":
    main()
