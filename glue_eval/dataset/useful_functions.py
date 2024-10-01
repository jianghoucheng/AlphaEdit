import pickle

def save_data(filename, data):
    #Storing data with labels
    a_file = open(filename, "wb")
    pickle.dump(data, a_file)
    a_file.close()
    

def load_data(filename):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output

def load_data_split(filename, split):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output[:split], output[split:]