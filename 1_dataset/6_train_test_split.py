import os
import random
import re
import numpy as np


with open("dataset/abstracts.csv") as f:
    abstracts = f.read().strip("\n").split("\n")

with open("dataset/authors.txt") as f:
    authors = f.read().strip("\n").split("\n")

with open("dataset/author2id.csv") as f:
    author2id = f.read().strip("\n").split("\n")

with open("dataset/content.csv") as f:
    content = f.read().strip("\n").split("\n")

with open("dataset/features.csv") as f:
    features = np.array([line.split(" ") for line in f.read().strip("\n").split("\n")], dtype=int)

with open("dataset/paper_ids.csv") as f:
    paper_ids = f.read().strip("\n").split("\n")

with open("dataset/references.csv") as f:
    references = f.read().strip("\n").split("\n")


assert(len(authors) == len(author2id))
assert(features.shape[0] == len(paper_ids))

train_ids = []
test_ids = []
for ids in author2id:
    ids = ids.strip(" ").split(" ")
    n_train = round(0.8*len(ids))
    random.shuffle(ids)
    train_ids.extend(ids[:n_train])
    test_ids.extend(ids[n_train:])

train_ids = list(set(train_ids))
test_ids = list(set(test_ids))
test_ids = [x for x in test_ids if x not in train_ids]
random.shuffle(train_ids)
random.shuffle(test_ids)
print(len(train_ids)/(len(train_ids) + len(test_ids)))
assert(abs(len(train_ids)/(len(train_ids) + len(test_ids)) - 0.8) < 0.05)

print("First paper in test set %s" % test_ids[0])
with open("dataset/split_id.txt", "w") as f:
    f.write(test_ids[0] + "\n")

with open("dataset/split_index.txt", "w") as f:
    f.write(str(len(train_ids)) + "\n")


ordered_paper_ids = train_ids.copy()
ordered_paper_ids.extend(test_ids)
print(len(ordered_paper_ids), len(train_ids), len(test_ids))

try:
    assert(len(ordered_paper_ids) == len(set(ordered_paper_ids)))
except AssertionError:
    seen = set()
    dupes = [x for x in ordered_paper_ids if x in seen or seen.add(x)]
    print(dupes)
    exit()

try:
    assert(set(ordered_paper_ids) == set(paper_ids)), \
        "We lost some paper here. That is weird."
except:
    notIn1 = [x for x in ordered_paper_ids if x not in paper_ids]
    notIn2 = [x for x in paper_ids if x not in ordered_paper_ids]
    print(notIn1)
    print(notIn2)
    exit()



lookup_index = {}
for idx, paper_id in enumerate(paper_ids):
    lookup_index[paper_id] = idx

paper_ids_index = []
for paper_id in ordered_paper_ids:
    try:
        paper_ids_index.append(lookup_index[paper_id])
    except:
        pass

assert(len(paper_ids_index) == len(paper_ids))

coauthors = np.where(np.sum(features, axis=1) > 1)[0]

# If a paper was coauthored by multiple authors, randomly assign it to one
# if the paper is a training set but leave multi-author cases for the
# evaluation set.
for idx in coauthors:
    if paper_ids[idx] not in train_ids:
        continue
    author_idx = np.where(features[idx,:] == 1)[0]
    select_author = np.random.choice(author_idx, 1)
    for author in author_idx:
        if author==select_author:
            continue
        else:
            author2id[author] = re.sub("\s*" + paper_ids[idx] + "\s*", 
                                       " ", 
                                       author2id[author])
            features[idx][author] = 0

for i in range(len(authors)):
    author2id[i] = author2id[i].strip(" ")

# Use the index list to reorder all the papers
with open("dataset/abstracts.csv", "w") as f:
    f.write("\n".join([abstracts[i] for i in paper_ids_index]) + "\n")

with open("dataset/author2id.csv", "w") as f:
    f.write("\n".join(author2id) + "\n")

with open("dataset/content.csv", "w") as f:
    f.write("\n".join([content[i] for i in paper_ids_index]) + "\n")

with open("dataset/paper_ids.csv", "w") as f:
    f.write("\n".join(ordered_paper_ids) + "\n")

with open("dataset/references.csv", "w") as f:
    f.write("\n".join([references[i] for i in paper_ids_index]) + "\n")

np.savetxt("dataset/features.csv", features[np.array(paper_ids_index),:], fmt="%1i")
