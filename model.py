import torch.nn
from transformers import BertModel


class BertForAuthorshipIdentification(torch.nn.Module):
    def __init__(self, droupout=0.3, checkpoint='bert-base-cased'):
        super(BertForAuthorshipIdentification, self).__init__()
        self.l1 = BertModel.from_pretrained(checkpoint)
        self.l2 = torch.nn.Dropout(droupout)
        self.l3 = torch.nn.Linear(768, 3)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output