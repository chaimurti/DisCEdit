import numpy as np

def parse_line(line):
    """ Parse a single line of the file and return structured data. """
    try:
        # Remove unnecessary characters and split the line by commas
        parts = line.strip().replace(' ', '').replace('[', '').replace(']', '').split(',')
        # Check if all expected parts are present
        if len(parts) < 5:  # Less than 5 parts indicates missing data
            print(f"Skipping incomplete line: {line}")
            return None

        # Create a dictionary from the parts
        data = {}
        for part in parts:
            key, value = part.split('=')
            data[key] = float(value)

        return data
    except Exception as e:
        print(f"Error parsing line: {line}. Error: {e}")
        return None

def read_and_process(filename):
    """ Read the file and process the data into the desired structure. """
    # Initialize a dictionary to hold the data arrays
    struct = {i: [] for i in range(10)}

    # Read file line by line
    with open(filename, 'r') as file:
        for line in file:
            data = parse_line(line)
            if data:
                c_id = int(data['c_id'])
                cacc_diff = data['cacc_new'] - data['cacc_old']
                ncacc_diff = data['ncacc_new'] - data['ncacc_old']
                pnum = data['pnum']
                
                # Append the data to the appropriate c_id in the structure
                struct[c_id].append([cacc_diff, ncacc_diff, pnum])

    # Convert lists to numpy arrays
    for key in struct:
        struct[key] = np.array(struct[key])

    return struct

# Example usage
filename = 'vgg16c10accs2.txt'
data_structure = read_and_process(filename)
print(data_structure)
for i in range(10):
    t =data_structure[i]
    print(i, t.shape)
    print(t[:,0])
    print(t[:,1])
    print(t[:,2])
# Include the pickle save function here
import pickle

def save_data_to_pickle(data, filename):
    """ Save the data to a pickle file. """
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

# Save the data structure
save_data_to_pickle(data_structure, 'vgg16c10accs2.pkl')
