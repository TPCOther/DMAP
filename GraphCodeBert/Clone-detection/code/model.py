import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np

def get_extended_attention_mask(attention_mask, input_shape, device = None, dtype = None):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
        return extended_attention_mask

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        self.query = 0
    
    def explain_forward(self, inputs_ids, attn_mask, position_idx, inputs_embeds=None):
        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        
        if inputs_embeds is None:
            outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]
        else:
            outputs = self.encoder.roberta(inputs_embeds=inputs_embeds,attention_mask=attn_mask,position_ids=position_idx)[0]

        logits=self.classifier(outputs)
        prob=F.softmax(logits)
        return prob
    
    def forward(self, inputs_ids_1,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2,labels=None, output_attentions = False): 
        bs,l=inputs_ids_1.size()
        inputs_ids=torch.cat((inputs_ids_1.unsqueeze(1),inputs_ids_2.unsqueeze(1)),1).view(bs*2,l)
        position_idx=torch.cat((position_idx_1.unsqueeze(1),position_idx_2.unsqueeze(1)),1).view(bs*2,l)
        attn_mask=torch.cat((attn_mask_1.unsqueeze(1),attn_mask_2.unsqueeze(1)),1).view(bs*2,l,l)

        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        
        if output_attentions:
            outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx, output_attentions = output_attentions)
            return outputs.attentions
        else:
            outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]
        # some errors comes from the update of transformers, so I changed it to the following line
        # pls refer to https://github.com/microsoft/CodeBERT/issues/73
        # outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx, token_type_ids = position_idx.eq(-1).long())[0]
        logits=self.classifier(outputs)
        prob=F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
    
    def get_results(self, dataset, batch_size, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=0,pin_memory=False)

        self.eval()
        logits=[] 
        for batch in eval_dataloader:
            (inputs_ids_1,position_idx_1,attn_mask_1,
            inputs_ids_2,position_idx_2,attn_mask_2,
            label)=[x.to("cuda")  for x in batch]
            with torch.no_grad():
                logit = self.forward(inputs_ids_1,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2)
                logits.append(logit.cpu().numpy())
                # 和defect detection任务不一样，这个的输出就是softmax值，而非sigmoid值

        logits=np.concatenate(logits,0)

        probs = logits
        pred_labels = [0 if first_softmax  > threshold else 1 for first_softmax in logits[:,0]]
        # 如果logits中的一个元素，其一个softmax值 > threshold, 则说明其label为0，反之为1

        return probs, pred_labels
    
    def get_attentions_output(self, dataset, layer, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=0,pin_memory=False)

        self.eval()
        attentions = []
        attention_masks = []
        attention_outputs = []
        for batch in eval_dataloader:
            (inputs_ids_1,position_idx_1,attn_mask_1,
            inputs_ids_2,position_idx_2,attn_mask_2,
            label)=[x.to("cuda")  for x in batch]

            bs,l=inputs_ids_1.size()
            inputs_ids=torch.cat((inputs_ids_1.unsqueeze(1),inputs_ids_2.unsqueeze(1)),1).view(bs*2,l)
            position_idx=torch.cat((position_idx_1.unsqueeze(1),position_idx_2.unsqueeze(1)),1).view(bs*2,l)
            attn_mask=torch.cat((attn_mask_1.unsqueeze(1),attn_mask_2.unsqueeze(1)),1).view(bs*2,l,l)

            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)        
            inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
            attention_mask =  get_extended_attention_mask(attn_mask, inputs_embeddings.size()[:-1], inputs_embeddings.device) 

            with torch.no_grad():
                attention = self.encoder.roberta.embeddings(position_ids=position_idx, token_type_ids=position_idx.eq(-1).long(),inputs_embeds=inputs_embeddings)
                for i in range(layer):
                    attention = self.encoder.roberta.encoder.layer[i](attention, attention_mask)[0]
                attention_output = self.encoder.roberta.encoder.layer[layer].attention(attention, attention_mask)[0]
                attention_outputs.append(attention_output)
                attentions.append(attention)
                attention_masks.append(attention_mask)

        return torch.cat(attentions, 0), torch.cat(attention_masks, 0), torch.cat(attention_outputs, 0)
    
    def continue_forward(self, data, attention_masks, layer, batch_size, threshold=0.5):
        self.eval()
        
        # data = data.to("cuda")
        # attention_masks = attention_masks.to("cuda")
        self.query += len(data) / 2
        logits = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            attention_mask = attention_masks[i:i+batch_size]
            with torch.no_grad():
                for i in range(layer, 12):
                    batch = self.encoder.roberta.encoder.layer[i](batch, attention_mask)[0]
                batch = self.classifier(batch)
                prob = F.softmax(batch, dim=-1)
            logits.append(prob.cpu().numpy())

        probs = np.concatenate(logits, 0)
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in probs[:, 0]]

        return probs, pred_labels
        

       