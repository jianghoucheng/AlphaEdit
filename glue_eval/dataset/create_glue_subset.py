from datasets import load_dataset
from useful_functions import save_data
import random

random.seed(37)


classwise_size = 100
# for dataset_name in ['sst2', 'mrpc', 'cola', 'rte']:
#     dataset = load_dataset("glue", dataset_name)
#     eval_dataset = dataset['validation']

#     classwise = {}
#     finalized_subset = []

#     # append examples to list for each label
#     for example in eval_dataset:
#         if example['label'] not in classwise:
#             classwise[example['label']] = [example]
#         else:
#             classwise[example['label']].append(example)
    

#     for label in classwise:
#         random.shuffle(classwise[label])
#         finalized_subset += classwise[label][:classwise_size]
    
#     #finalized_subset = classwise['yes'] + classwize['no']

#     random.shuffle(finalized_subset)
#     save_data(dataset_name + '.pkl', finalized_subset)


dataset = load_dataset("glue", "sst2")
eval_dataset = dataset['validation']

classwise = {}
finalized_subset = []

# append examples to list for each label
for example in eval_dataset:
    if example['label'] not in classwise:
        classwise[example['label']] = [example]
    else:
        classwise[example['label']].append(example)

# shuffle examples within each label
for label in classwise:
    random.shuffle(classwise[label])

classwise_size = min(len(examples) for examples in classwise.values())

# Prepare the finalized subset with alternating rows of classes
index = 0
while len(finalized_subset) < classwise_size * len(classwise):
    for label in classwise:
        if index < len(classwise[label]):
            finalized_subset.append(classwise[label][index])
    index += 1

# Save the finalized subset
save_data('sst2.pkl', finalized_subset)