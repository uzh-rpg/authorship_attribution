""" Summary
This script is somewhat the same as the 3_postprocess_data.py script,
with the only exception that the decision whether a paper must be removed
is solely based on the author.
If the 4_seperate_authors.py script found that it is likely that multiple
authors (persons) are associated with the same name, the author will be
removed alltogether.
Also, in the rare case that a paper is jointly authored by more than 1
author in the list, it is randomly assigned to one of the coauthors.
"""

import numpy as np
from itertools import compress

# Read all files
with open("dataset/abstracts.csv") as f:
    abstracts = f.read().strip("\n").split("\n")

with open("dataset/author2id.csv") as f:
    author2id = f.read().strip("\n").split("\n")

with open("dataset/authors.txt") as f:
    authors = f.read().strip("\n").split("\n")

with open("dataset/content.csv") as f:
    content = f.read().strip("\n").split("\n")

with open("dataset/features.csv") as f:
    features = np.array([line.split(" ") for line in f.read().strip("\n").split("\n")], dtype=int)

with open("dataset/paper_ids.csv") as f:
    paper_ids = f.read().strip("\n").split("\n")

with open("dataset/references.csv") as f:
    references = f.read().strip("\n").split("\n")

with open("dataset/bad_authors.csv") as f:
    bad_authors = f.read().strip("\n").split("\n")

assert(len(author2id) == len(authors))
assert(len(author2id) == len(features[0]))
idx_good = [i for (i,a) in enumerate(authors) if a not in bad_authors]

# Find bad authors and create a mask
features = features[:,idx_good]
authors = [authors[i] for i in idx_good]
author2id = [author2id[i] for i in idx_good]

mask = np.sum(features, axis=1)!=0
features = features[mask,:]

# Write back
with open("dataset/authors.txt", "w") as f:
    f.write("\n".join(authors) + "\n")

with open("dataset/abstracts.csv", "w") as f:
    f.write("\n".join(list(compress(abstracts, mask))) + "\n")

with open("dataset/author2id.csv", "w") as f:
    f.write("\n".join(author2id) + "\n")

with open("dataset/content.csv", "w") as f:
    f.write("\n".join(list(compress(content, mask))) + "\n")

with open("dataset/paper_ids.csv", "w") as f:
    f.write("\n".join(list(compress(paper_ids, mask))) + "\n")

with open("dataset/references.csv", "w") as f:
    f.write("\n".join(list(compress(references, mask))) + "\n")

np.savetxt("dataset/features.csv", features, fmt="%1i")
