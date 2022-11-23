import os
import numpy as np

# Read in all files and split them according to the specification.
with open("dataset/abstracts.csv") as f:
    abstracts = f.read().strip("\n").split("\n")

with open("dataset/content.csv") as f:
    content = f.read().strip("\n").split("\n")

with open("dataset/features.csv") as f:
    features = f.read().strip("\n").split("\n")

with open("dataset/paper_ids.csv") as f:
    paper_ids = f.read().strip("\n").split("\n")

with open("dataset/references.csv") as f:
    references = f.read().strip("\n").split("\n")

with open("dataset/reference_authors.csv") as f:
    reference_authors = f.read().strip("\n").split("\n")

with open("dataset/reference_count_embeddings.csv") as f:
    reference_embeddings = f.read().strip("\n").split("\n")


# Quick sanity check if the lengths all match up
assert(len(abstracts) == len(content))
assert(len(content) == len(features))
assert(len(features) == len(paper_ids))
assert(len(paper_ids) == len(references))
assert(len(references) == len(reference_authors))
assert(len(reference_embeddings) == len(reference_authors))

token_length = 512
indices = []
new_content = []
for i in range(0,len(content)): # EOF is \n thus one too many
    text = content[i]
    text_split = text.split(" ")
    text_join = [" ".join(text_split[l:l+token_length])
            for l in range(0, len(text_split), token_length)]
    if len(text_join) > 1:
        text_join = text_join[:-1]
    if len(text_join) > 1:
        text_join = [x for x in text_join if len(x)/x.count(" ") > 4.22]

    new_content.extend(text_join)

    indices.extend(len(text_join)*[i])

assert(len(indices) == len(new_content))

new_features = []
new_paper_ids = []
new_references = []
new_reference_authors = []
new_references_embeddings = []
new_abstracts = []
for idx in indices:
    new_features.append(features[idx])
    new_paper_ids.append(paper_ids[idx])
    new_references.append(references[idx])
    new_abstracts.append(abstracts[idx])
    new_reference_authors.append(reference_authors[idx])
    new_references_embeddings.append(reference_embeddings[idx])

assert(len(new_abstracts) == len(new_content))
assert(len(new_content) == len(new_features))
assert(len(new_features) == len(new_paper_ids))
assert(len(new_paper_ids) == len(new_references))
assert(len(new_references) == len(new_reference_authors))
assert(len(new_references_embeddings) == len(new_reference_authors))

# Write back
with open("dataset/abstracts.csv", "w") as f:
    f.write("\n".join(new_abstracts) + "\n")

with open("dataset/content.csv", "w") as f:
    f.write("\n".join(new_content) + "\n")

with open("dataset/paper_ids.csv", "w") as f:
    f.write("\n".join(new_paper_ids) + "\n")

with open("dataset/references.csv", "w") as f:
    f.write("\n".join(new_references) + "\n")

with open("dataset/reference_authors.csv", "w") as f:
    f.write("\n".join(new_reference_authors) + "\n")

with open("dataset/reference_count_embeddings.csv", "w") as f:
    f.write("\n".join(new_references_embeddings) + "\n")

with open("dataset/features.csv", "w") as f:
    f.write("\n".join(new_features) + "\n")

with open("dataset/split_id.txt", "r") as f:
    split_id = f.read().strip("\n").strip(" ")

with open("dataset/split_index.txt", "w") as f:
    f.write(str(new_paper_ids.index(split_id)) + "\n")

with open("dataset/reference_authors.csv") as f:
    index = [i for i,x in enumerate(f.read().strip("\n").splitlines()) if x.strip() == ""]

with open("dataset/reference_null.csv", "w") as f:
    f.write("\n".join([str(i) for i in index]) + "\n")
