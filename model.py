import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import *
from pytorch_transformers.modeling_bert import *

class RelationClassifer(BertPreTrainedModel):

    def __init__(self, config):
        super(RelationClassifer, self).__init__(config)
#       super().__init__(config)
        self.num_labels = 3

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.apply(self.init_weights)

#       self.init_weights()


    def forward(
        self, 
        input_ids, 
        token_type_ids=None, 
        attention_mask=None, 
        entity1_mask=None, 
        entity2_mask=None, 
        labels=None):
        
        with torch.no_grad():

            # Get the last output layer of BERT        
            encoded_layers = self.bert(input_ids, attention_mask)[0]
            batch_size, max_seq_length = entity1_mask.shape[0], entity1_mask.shape[1]

            # Get the corresponding embedding of E1 and E2
            diag_entity1_mask_ = []
            for i in range(batch_size):
                diag_entity1_mask_.append(torch.diag(entity1_mask[i]).cpu().numpy())
            diag_entity1_mask = torch.tensor(diag_entity1_mask_).cuda()

            diag_entity2_mask_ = []
            for i in range(batch_size):
                diag_entity2_mask_.append(torch.diag(entity2_mask[i]).cpu().numpy())
            diag_entity2_mask = torch.tensor(diag_entity2_mask_).cuda()

            # Concatenate two entity embedding      
            batch_entity1_emb = torch.matmul(diag_entity1_mask, encoded_layers).permute(0,2,1)
            batch_entity2_emb = torch.matmul(diag_entity2_mask, encoded_layers).permute(0,2,1)
            batch_entity_emb = torch.cat((batch_entity1_emb, batch_entity2_emb), dim=1)


            pooling = nn.MaxPool1d(kernel_size=max_seq_length, stride=1)
            entity_emb_output = pooling(batch_entity_emb).squeeze()
            entity_emb_output = self.dropout(entity_emb_output)
        
        # Linear layer classifier
        logits = self.classifier(entity_emb_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return logits , loss
        else:
            return logits

        return outputs  

