# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
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

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float16).min
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
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args=args
        self.query = 0
    
        
    def forward(self, input_ids=None, labels=None, inputs_embeds=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        if inputs_embeds is not None:
            outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=input_ids.ne(1))[0]
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size):
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0, pin_memory=False)

        eval_loss = 0.0
        nb_eval_steps = 0
        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")       
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs,label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
            nb_eval_steps += 1
        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))

        return probs, pred_labels
    
    def get_attentions_output(self, dataset, layer, batch_size):
        # 提取探针注意力时开启
        # dataset = [[torch.tensor(x)] for x in dataset]
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=0,pin_memory=False)

        self.eval()
        attentions = []
        attention_masks = []
        attention_outputs = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            attention_mask = inputs.ne(1)
            attention_mask = get_extended_attention_mask(attention_mask, inputs.size(), inputs.device)
            with torch.no_grad():
                attention = self.encoder.embeddings(inputs)
                for i in range(layer):
                    attention = self.encoder.encoder.layer[i](attention, attention_mask)[0]
                attention_output = self.encoder.encoder.layer[layer].attention(attention, attention_mask)[0]
                attention_outputs.append(attention_output)
                attentions.append(attention)
                attention_masks.append(attention_mask)

        return torch.cat(attentions, 0), torch.cat(attention_masks, 0), torch.cat(attention_outputs, 0)
    
    def continue_forward(self, data, attention_masks, layer, batch_size):
        self.eval()
        
        # data = data.to("cuda")
        # attention_masks = attention_masks.to("cuda")

        logits = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            attention_mask = attention_masks[i:i+batch_size]
            self.query += len(batch)
            with torch.no_grad():
                for i in range(layer, 12):
                    batch = self.encoder.encoder.layer[i](batch, attention_mask)[0]
                batch = self.classifier(batch)
                prob = F.softmax(batch, dim=-1)
            logits.append(prob.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = [np.argmax(label) for label in logits]

        return probs, pred_labels
    
    
        
 