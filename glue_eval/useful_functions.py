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

# def load_data_split(filename, split):
#     a_file = open(filename, "rb")
#     output = pickle.load(a_file)
#     a_file.close()
#     return output[:split], output[split:]

FEW_SHOT_TEST_SPLIT = 10

def load_data_split(filename, number_of_few_shots, number_of_tests):
    a_file = open(filename, "rb")
    output = pickle.load(a_file)
    a_file.close()
    assert number_of_few_shots <= FEW_SHOT_TEST_SPLIT, f"The largest number of few shot can only be 100, we received {number_of_few_shots}"
    if not number_of_tests is None:
        assert number_of_tests <= len(output) - FEW_SHOT_TEST_SPLIT,  f"The largest number of test for this task can only be {len(output) - FEW_SHOT_TEST_SPLIT}, we received {number_of_tests}"
    else:
        number_of_tests = len(output) - FEW_SHOT_TEST_SPLIT
    allow_few_shots, allow_tests = output[:FEW_SHOT_TEST_SPLIT], output[FEW_SHOT_TEST_SPLIT:]
    return allow_few_shots[:number_of_few_shots], allow_tests[:number_of_tests]

MODEL_NAME_TO_MAXIMUM_CONTEXT_LENGTH_MAP = {
    "gpt2-xl": 1024,
    "llama-2-7b-hf": 4096,
    "llama3-8b-instruct": 4096,
    "eleutherai_gpt-j-6b": 2048,
    "gpt2-large": 1024,
    "gpt2-medium": 1024
}