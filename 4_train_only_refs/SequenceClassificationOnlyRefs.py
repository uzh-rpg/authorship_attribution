import torch
from torch import nn
import transformers
from transformers import activations
from transformers import DistilBertTokenizer, \
DistilBertForSequenceClassification, DistilBertModel, DistilBertPreTrainedModel, Trainer, TrainingArguments,EvalPrediction, AutoTokenizer
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

class SequenceClassificationOnlyRefs(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # self.distilbert = DistilBertModel(self.config)

    def initialize(self, ref_count_embedding_size):
        self.ref_count_embedding_size = ref_count_embedding_size  # 1153 for D500, 2107 for D400, 5290 for D300 and 12414 for D200
        self.ref_concat_size = 128     # hardcoded for now
        self.ref_mid_size = int((self.ref_count_embedding_size + self.ref_concat_size) / 2)

        self.ref_embedding_1 = nn.Linear(self.ref_count_embedding_size, self.ref_mid_size)
        self.ref_embedding_2 = nn.Linear(self.ref_mid_size, self.ref_concat_size)
        self.pre_classifier = nn.Linear(# self.config.dim +
                                        self.ref_concat_size, self.config.dim)
        self.classifier = nn.Linear(self.config.dim, self.config.num_labels)
        self.dropout = nn.Dropout(self.config.seq_classif_dropout)

        # # Initialize weights and apply final processing
        # self.post_init()

    # def get_position_embeddings(self) -> nn.Embedding:
    #     """
    #     Returns the position embeddings
    #     """
    #     return self.distilbert.get_position_embeddings()

    # def resize_position_embeddings(self, new_num_position_embeddings: int):
    #     """
    #     Resizes position embeddings of the model if :obj:`new_num_position_embeddings !=
    #     config.max_position_embeddings`.
    #     Arguments:
    #         new_num_position_embeddings (:obj:`int`):
    #             The number of new position embedding matrix. If position embeddings are learned, increasing the size
    #             will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
    #             end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
    #             size will add correct vectors at the end following the position encoding algorithm, whereas reducing
    #             the size will remove vectors from the end.
    #     """
    #     self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        reference_count_embeddings=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # distilbert_output = self.distilbert(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        ref_embedding_output = self.ref_embedding_1(reference_count_embeddings.type(torch.cuda.HalfTensor))
        ref_embedding_output = nn.ReLU()(ref_embedding_output)
        ref_embedding_output = self.dropout(ref_embedding_output)
        ref_embedding_output = self.ref_embedding_2(ref_embedding_output)
        ref_embedding_output = nn.ReLU()(ref_embedding_output)
        ref_embedding_output = self.dropout(ref_embedding_output)
        # hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        # tuple_cat = (hidden_state[:,0], ref_embedding_output)
        # pooled_output = torch.cat(tuple_cat, dim=1)  # (bs, dim)
        pooled_output = ref_embedding_output
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # if not return_dict:
        #     output = (logits,) + distilbert_output[1:]
        #     return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=distilbert_output.hidden_states,
            # attentions=distilbert_output.attentions,
        )
