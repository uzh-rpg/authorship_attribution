""" Summary
Not all papers that have been selected by the 1_create_dataset.py script
have a meaningful content/reference section extracted by the second script.
In this third step, all such papers are removed from the datset. Note that
this also means that even if an author originally had >=N papers in the
dataset, this number is reduced now on average by about 1/3.

Logic: If any of the entries in
    abstracts.csv
    content.csv
    references.csv
for a given paper is blank, it is removed from the dataset.
"""

import os
import re
from itertools import compress

# Read in all files and split them according to the specification.
with open("dataset/abstracts.csv") as f:
    abstracts = f.read().split("\n")

with open("dataset/author2id.csv") as f:
    author2id = f.read()

with open("dataset/content.csv") as f:
    content = f.read().split("\n")

with open("dataset/features.csv") as f:
    features = f.read().split("\n")

with open("dataset/paper_ids.csv") as f:
    paper_ids = f.read().split("\n")

with open("dataset/references.csv") as f:
    references = f.read().split("\n")

# Quick sanity check if the lengths all match up
assert(len(abstracts) == len(content))
assert(len(content) == len(features))
assert(len(features) == len(paper_ids))
assert(len(paper_ids) == len(references))

N = len(abstracts)
print("Found %i entries" % N)

# Create masks that are true if the entries in all files are valid
mask_abstracts = [bool(l.strip()) for l in abstracts]
mask_content = [bool(l.strip()) for l in content]
mask_features = [bool(l.strip()) for l in features]
mask_ids = [bool(l.strip()) for l in paper_ids]
mask_references = [bool(l.strip()) for l in references]


mask = [True]*N
for i in range(N):
    mask[i] &= mask_abstracts[i]
    mask[i] &= mask_content[i]
    mask[i] &= mask_features[i]
    mask[i] &= mask_ids[i]
    mask[i] &= mask_references[i]

bad_ids = []
for i in range(N):
    if not mask[i]:
        bad_ids.append(paper_ids[i])

print("Found %i bad entries" % len(bad_ids))

# Write the files again, but mask the invalid entries.
for paper_id in bad_ids:
    author2id = author2id.replace(paper_id, "")

with open("dataset/abstracts.csv", "w") as f:
    f.write("\n".join(list(compress(abstracts, mask))) + "\n")

with open("dataset/author2id.csv", "w") as f:
    f.write(author2id)

with open("dataset/content.csv", "w") as f:
    f.write("\n".join(list(compress(content, mask))) + "\n")

with open("dataset/paper_ids.csv", "w") as f:
    f.write("\n".join(list(compress(paper_ids, mask))) + "\n")

with open("dataset/references.csv", "w") as f:
    f.write("\n".join(list(compress(references, mask))) + "\n")

with open("dataset/features.csv", "w") as f:
    f.write("\n".join(list(compress(features, mask))) + "\n")
