import torch.nn
from transformers import BertModel, AutoConfig
from transformers.models.bert.modeling_bert import BertPooler
from model_params import MEAN, POOLING, CLS


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BertSingleAV(torch.nn.Module):
    def __init__(self, dropout=0.3, checkpoint="bert-base-cased", pooling_method=MEAN):
        super(BertSingleAV, self).__init__()

        self.pooling_method = pooling_method

        # output of bert
        self.l1 = BertModel.from_pretrained(checkpoint)
        self.dropout = torch.nn.Dropout(p=dropout)
        # input is bert pooler, this acts as an embedding layer
        self.l2 = torch.nn.Linear(in_features=768, out_features=512, bias=False)

    def forward(self, ids, mask):
        out = self.l1(ids, mask)
        if self.pooling_method == CLS:
            pooled_out = out.pooler_output
        elif self.pooling_method == MEAN:
            pooled_out = mean_pooling(out.last_hidden_state, mask)
        else:
            raise Exception

        pooled_out_after_dropout = self.dropout(pooled_out)
        embedding = self.l2(pooled_out_after_dropout)
        return embedding


class BertSiamAV(torch.nn.Module):
    def __init__(self, dropout=0.3, checkpoint="bert-base-cased", pooling_method=MEAN):
        super(BertSiamAV, self).__init__()
        # use the same bert for the bi-encoder, rather than 2 with shared weights
        # 2 individual berts won't fit on a single gpu
        self.subnet1 = BertSingleAV(dropout, checkpoint, pooling_method)
        self.subnet2 = self.subnet1

        # activation? No activation - similar to BertForSequenceClassification
        self.classifier = torch.nn.Linear(in_features=512 * 3, out_features=2)

    def forward(self, ids1, mask1, ids2, mask2):
        embedding1 = self.subnet1(ids1, mask1)
        embedding2 = self.subnet2(ids2, mask2)
        # the classification objective function from https://arxiv.org/pdf/1908.10084.pdf
        # todo check
        concatenated_embedding = torch.cat((embedding1, embedding2, torch.sub(embedding1, embedding2)), dim=1)
        logits = self.classifier(concatenated_embedding)
        return logits

    def freeze_subnetworks(self):
        # l1 in the subnetwork corresponds to bert
        for parameter in self.subnet1.l1.parameters():
            parameter.requires_grad = False

    def unfreeze_subnetworks(self):
        # l1 in the subnetwork corresponds to bert
        for parameter in self.subnet1.l1.parameters():
            parameter.requires_grad = True

