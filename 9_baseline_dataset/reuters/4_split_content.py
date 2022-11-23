import os
import numpy as np

# Read in all files and split them according to the specification.
with open("dataset/content.csv") as f:
    content = f.read().strip("\n").split("\n")

with open("dataset/features.csv") as f:
    features = f.read().strip("\n").split("\n")

with open("dataset/paper_ids.csv") as f:
    paper_ids = f.read().strip("\n").split("\n")

# Quick sanity check if the lengths all match up
assert(len(content) == len(features))
assert(len(features) == len(paper_ids))

token_length = 512
indices = []
new_content = []
for i in range(0,len(content)): # EOF is \n thus one too many
    text = content[i]
    text_split = text.split(" ")
    text_join = [" ".join(text_split[l:l+token_length]) 
            for l in range(0, len(text_split), token_length)]
    if len(text_join) > 1:
        text_join = [text for text in text_join if len(text) > 250]

    new_content.extend(text_join)
    indices.extend(len(text_join)*[i])

assert(len(indices) == len(new_content))

new_features = []
new_paper_ids = []
for idx in indices:
    new_features.append(features[idx])
    new_paper_ids.append(paper_ids[idx])

assert(len(new_content) == len(new_features))
assert(len(new_features) == len(new_paper_ids))

# Write back
with open("dataset/content.csv", "w") as f:
    f.write("\n".join(new_content) + "\n")

with open("dataset/paper_ids.csv", "w") as f:
    f.write("\n".join(new_paper_ids) + "\n")

with open("dataset/features.csv", "w") as f:
    f.write("\n".join(new_features) + "\n")

with open("dataset/split_id.txt", "r") as f:
    split_id = f.read().strip("\n").strip(" ")

with open("dataset/split_index.txt", "w") as f:
    f.write(str(new_paper_ids.index(split_id)) + "\n")
