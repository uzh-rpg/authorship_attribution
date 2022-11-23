import torch
from torch import nn
torch.cuda.empty_cache()
import datasets
import transformers
from transformers import activations
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, \
DistilBertForSequenceClassification, DistilBertModel, DistilBertPreTrainedModel, Trainer, TrainingArguments,EvalPrediction, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, hamming_loss
import wandb
import random
import sys
import os
from SequenceClassificationOnlyRefs import SequenceClassificationOnlyRefs

# ==========================================  MAIN PARAMETERS START ==========================================
# API_KEY for good results, not for debuging. Use responsibly, or comment the line out!
# os.environ['WANDB_API_KEY'] = 'ba494c0f921b281d1e02646777f2c50d4181184a'

RUN_NAME = "Ref_D500C_5e-5"
DATASET_PARENT_FOLDER ='/data/storage/bauersfeld/'
DATASET_BASE_FOLDER = 'dataset-500'
CHUNKED_DATASET = True         # if the dataset comes in chunks of 512
if (CHUNKED_DATASET):
    DATASET_BASE_FOLDER = DATASET_BASE_FOLDER + '-split'
N_EPOCHS = 10
LEARNING_RATE = 5e-5
desired_num_eval_steps_per_epoch = 6
# ===========================================  MAIN PARAMETERS END ============================================

# SECONDARY PARAMETERS
MAX_LENGTH = 512
LIM_NUM_CHUNKS = 0              # set to 0 for no limit, which will take into account all chunks
SPLIT_INDEX = True

# save path
SAVE_PATH = "./results/" + RUN_NAME + "_result"
data_path_content = os.path.join(DATASET_PARENT_FOLDER, DATASET_BASE_FOLDER, 'content.csv')
data_path_features = os.path.join(DATASET_PARENT_FOLDER, DATASET_BASE_FOLDER, 'features.csv')
data_path_paper_ids = os.path.join(DATASET_PARENT_FOLDER, DATASET_BASE_FOLDER, 'paper_ids.csv')
data_path_reference_count_embeddings = os.path.join(DATASET_PARENT_FOLDER, DATASET_BASE_FOLDER, 'reference_count_embeddings.csv')

# LOAD_PATH = "./results/Ref+Cont_D300C_result_dataset"
LOAD_TOKENIZED_DATA_FROM_FILE = False  # If True, does not run the tokenization/dataloading step, but takes it from SAVE_PATH
# LOAD_PATH = "./results/Ref+Cont_D300C_result_dataset"
LOAD_TOKENIZED_DATA_FROM_FILE = False  # If True, does not run the tokenization/dataloading step, but takes it from SAVE_PATH

if (LOAD_TOKENIZED_DATA_FROM_FILE):
    dataset_dict = datasets.load_from_disk(LOAD_PATH)
    train_data = dataset_dict['train']
    test_data = dataset_dict['test']

else:
    if SPLIT_INDEX:
        data_path_train_split_index = os.path.join(DATASET_PARENT_FOLDER, DATASET_BASE_FOLDER, 'split_index.txt')

        with open(data_path_train_split_index) as f:
            SPLIT_INDEX = int(f.readline())

    print(SAVE_PATH)

    labels = pd.read_csv(data_path_features, header=None, sep='\s+')
    paper_ids = pd.read_csv(data_path_paper_ids, header=None, sep='\s+')
    reference_count_embeddings = pd.read_csv(data_path_reference_count_embeddings, header=None, sep='\s+')
    print(reference_count_embeddings.shape)
    print(labels.shape)
    filtered_df = pd.DataFrame()
    filtered_df['paper_ids'] = paper_ids
    filtered_df['label_multi'] = labels.values.tolist()
    filtered_df['label_single'] = np.argmax(labels.values.tolist(), axis = 1)
    filtered_df['reference_count_embeddings'] = reference_count_embeddings.values.tolist()
    filtered_df['label'] = filtered_df['label_single']

    def drop_chunks(data, lim_num_chunks):
        curr_paper_id = 0
        rep_count = 0
        for idx, row in data.iterrows():
            paper_id_idx = row['paper_ids']

            if (curr_paper_id != paper_id_idx):
                rep_count = 0
                curr_paper_id = paper_id_idx
                # next iteration
            else:
                rep_count = rep_count + 1
                if (rep_count >= lim_num_chunks):
                    data.drop(idx, inplace = True)

    if (SPLIT_INDEX):
        train_data = filtered_df.iloc[:SPLIT_INDEX]
        test_data = filtered_df.iloc[SPLIT_INDEX:]

        if (CHUNKED_DATASET and LIM_NUM_CHUNKS != 0):
            drop_chunks(train_data, LIM_NUM_CHUNKS)
            drop_chunks(test_data, LIM_NUM_CHUNKS)

        train_data = datasets.Dataset.from_pandas(train_data)
        test_data = datasets.Dataset.from_pandas(test_data)

    else:
        total_data = datasets.Dataset.from_pandas(filtered_df)
        dataset_dic = total_data.train_test_split(test_size=0.2)
        train_data = dataset_dic['train']
        test_data = dataset_dic['test']


    dataset_to_save = datasets.DatasetDict({
        "train": train_data,
        "test": test_data
    })

    print("saving dataset...")
    dataset_to_save.save_to_disk(SAVE_PATH + "_dataset")

print(train_data)
print(test_data)
LABEL_SIZE = len(train_data['label_multi'][0])
REFERENCE_COUNT_EMBEDDINGS_SIZE = len(train_data['reference_count_embeddings'][0])
print("LABEL_SIZE")
print(LABEL_SIZE)
model = SequenceClassificationOnlyRefs.from_pretrained('distilbert-base-uncased', num_labels = LABEL_SIZE)
model.initialize(REFERENCE_COUNT_EMBEDDINGS_SIZE)
print(model)

# Report number of trainable parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of trainable parameters: ")
print(params)

# define accuracy metrics

def my_accuracy(pred_df):
    num_correct_preds = 0
    for idx, row in pred_df.iterrows():
        pred_idx = row['preds_argmax']
        label_multi = row['labels_multi']
        if (label_multi[pred_idx] == 1):
            num_correct_preds = num_correct_preds + 1

    return num_correct_preds / len(pred_df)

def compute_metrics(pred):
    labels = pred.label_ids

    # Get all unique paper ids in test set:
    original_df = pd.DataFrame(columns=['paper_ids', 'labels', 'labels_multi' ,'preds_raw', 'preds_argmax'])
    original_df['paper_ids'] = test_data['paper_ids']
    original_df['labels'] = test_data['label']
    original_df['labels_multi'] = test_data['label_multi']

    for idx in range(0, len(test_data)):
        original_df.at[idx, 'preds_raw'] = pred.predictions[idx]
        original_df.at[idx, 'preds_argmax'] = np.argmax(original_df['preds_raw'].iloc[idx])

    print(original_df)

    if (CHUNKED_DATASET):
        unique_id_df = pd.DataFrame(columns=['paper_ids', 'labels', 'labels_multi', 'preds_raw', 'preds_argmax'])
        unique_id_df['paper_ids'] = original_df['paper_ids'].unique()

        for idx, row in unique_id_df.iterrows():
            paper_id = row['paper_ids']
            mask_papers_same_id = (original_df['paper_ids'] == paper_id)
            unique_id_df.at[idx,'labels'] = original_df.loc[mask_papers_same_id]['labels'].values[0]
            unique_id_df.at[idx,'labels_multi'] = original_df.loc[mask_papers_same_id]['labels_multi'].values[0]
            unique_id_df.at[idx, 'preds_raw'] = original_df.loc[mask_papers_same_id]['preds_raw'].mean()
            unique_id_df.at[idx, 'preds_argmax'] = np.argmax(unique_id_df['preds_raw'].iloc[idx])

        print(unique_id_df)
        result_df = unique_id_df
    else:
        result_df = original_df

    preds_final = np.array(result_df['preds_argmax'].values, dtype=np.int32)
    labels_final = np.array(result_df['labels'].values, dtype=np.int32)

    acc = my_accuracy(result_df)
    return {
        'accuracy': acc
    }

training_batch_size = 32
gradient_accumulation_steps = 4

eval_steps = int(len(train_data) / training_batch_size / gradient_accumulation_steps / desired_num_eval_steps_per_epoch)
print("eval_steps")
print(eval_steps)

# define the training arguments
training_args = TrainingArguments(
    output_dir = SAVE_PATH + '/checkpoints/',
    num_train_epochs = N_EPOCHS,
    per_device_train_batch_size = training_batch_size,
    gradient_accumulation_steps = 4,
    per_device_eval_batch_size= 64,
    disable_tqdm = False,
    evaluation_strategy ="steps",
    eval_steps = eval_steps, # Evaluation and Save happens every 50 steps
    # save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
    # load_best_model_at_end=True,
    warmup_steps=160,
    weight_decay=0.01,
    logging_steps = 4,
    learning_rate = LEARNING_RATE,
    fp16 = True,
    logging_dir = SAVE_PATH + '/logs/',
    dataloader_num_workers = 0,
    run_name = RUN_NAME
)
# instantiate the trainer class and check for available devices
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainer.train()

# save the best model
trainer.model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

trainer.evaluate()
