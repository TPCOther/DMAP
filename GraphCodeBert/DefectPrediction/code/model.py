# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import time
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.utils.data import SequentialSampler, DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np

class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.name} Elapsed time: {self.elapsed_time:.4f} seconds")

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
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
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
        self.classifier = RobertaClassificationHead(config)
        self.args=args
        self.query = 0
        
    def forward(self, inputs_ids=None, attn_mask=None, position_idx=None, labels = None, inputs_embeds=None):
        #embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)

        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
        if inputs_embeds is not None:
            inputs_embeddings = inputs_embeds
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
      
    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0, pin_memory=False)

        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs_ids = batch[0].to("cuda")
            attn_mask = batch[1].to("cuda")
            position_idx = batch[2].to("cuda")
            label = batch[3].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs_ids, attn_mask, position_idx, label)
                logits.append(logit.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels
    
    def get_attentions_output(self, dataset, layer, batch_size):
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=0,pin_memory=False)

        self.eval()
        attentions = []
        attention_masks = []
        attention_outputs = []
        for batch in eval_dataloader:
            inputs_ids = batch[0].to("cuda")       
            attn_mask = batch[1].to("cuda") 
            position_idx = batch[2].to("cuda") 
            # label=batch[3].to("cuda")

            nodes_mask=position_idx.eq(0)
            token_mask=position_idx.ge(2)
            inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
            nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
            attention=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
            attention_mask =  get_extended_attention_mask(attn_mask, attention.size()[:-1], attention.device)
        
            with torch.no_grad():
                attention = self.encoder.roberta.embeddings(position_ids=position_idx, token_type_ids=position_idx.eq(-1).long(),inputs_embeds=attention)
                for i in range(layer):
                    attention = self.encoder.roberta.encoder.layer[i](attention, attention_mask)[0]
                attention_output = self.encoder.roberta.encoder.layer[layer].attention(attention, attention_mask)[0]
                attention_outputs.append(attention_output)
                attentions.append(attention)
                attention_masks.append(attention_mask)
    
        attentions = torch.cat(attentions, 0)
        attention_masks = torch.cat(attention_masks, 0)
        attention_outputs = torch.cat(attention_outputs, 0)

        return attentions, attention_masks, attention_outputs
    
    def continue_forward(self, data, attention_masks, layer, batch_size):
        self.eval()

        logits = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            attention_mask = attention_masks[i:i+batch_size]
            self.query += len(batch)
            with torch.no_grad():
                for i in range(layer, 12):
                    batch = self.encoder.roberta.encoder.layer[i](batch, attention_mask)[0]
                batch = self.classifier(batch)
                prob = F.softmax(batch, dim=-1)
            logits.append(prob.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = [np.argmax(label) for label in logits]

        return probs, pred_labels