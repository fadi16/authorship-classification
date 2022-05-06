import torch.nn
from transformers import BertModel, AutoConfig
from transformers.models.bert.modeling_bert import BertPooler
from model_params import MEAN, POOLING, CLS


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class BertSingle(torch.nn.Module):
    def __init__(self, dropout=0.3, checkpoint="bert-base-cased", pooling_method=MEAN):
        super(BertSingle, self).__init__()

        self.pooling_method = pooling_method

        # output of bert
        self.l1 = BertModel.from_pretrained(checkpoint)
        self.dropout = torch.nn.Dropout(p=dropout)
        # input is bert pooler, this acts as an embedding layer
        self.l2 = torch.nn.Linear(in_features=768, out_features=256, bias=False)
        self.fn2 = torch.nn.Tanh()

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
        embedding = self.fn2(embedding)
        return embedding


class BertSiam(torch.nn.Module):
    def __init__(self, dropout=0.3, checkpoint="bert-base-cased", pooling_method=MEAN):
        super(BertSiam, self).__init__()
        # use the same bert for the bi-encoder, rather than 2 with shared weights
        # 2 individual berts won't fit on a single gpu
        self.subnet1 = BertSingle(dropout, checkpoint, pooling_method).to()
        self.subnet2 = self.subnet1
        self.cosine_sim = torch.nn.CosineSimilarity(dim=1)

    def forward(self, ids1, mask1, ids2, mask2):
        embedding1 = self.subnet1.forward(ids1, mask1)
        embedding2 = self.subnet2.forward(ids2, mask2)

        return embedding1, embedding2

    def get_embedding(self, ids, mask):
        return self.subnet1.forward(ids, mask)

    def freeze_subnetworks(self):
        # l1 in the subnetwork corresponds to bert
        for parameter in self.subnet1.l1.parameters():
            parameter.requires_grad = False

    def unfreeze_subnetworks(self):
        # l1 in the subnetwork corresponds to bert
        for parameter in self.subnet1.l1.parameters():
            parameter.requires_grad = True

    def save_pretrained(self, model_checkpoint_path):
        # the only thing that needs saving is the subnetwork - both are the same
        torch.save(self.subnet1.state_dict(), model_checkpoint_path)

    def load_fine_tuned_weights(self, model_checkpoint_path):
        state = torch.load(model_checkpoint_path)
        self.subnet1.load_state_dict(state)
