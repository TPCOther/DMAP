import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import SequentialSampler, DataLoader
from modeling_attn_mask_utils import AttentionMaskConverter
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

class CodeT5RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder.model.encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = CodeT5RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None, inputs_embeds=None, attention_mask=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        if inputs_embeds is not None:
            outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, output_hidden_states=True)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['hidden_states'][-1]
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        outputs = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                  hidden_states.size(-1))[:, -1, :]

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
    
    def get_attentions(self, dataset, batch_size):
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=0,pin_memory=False)
        self.eval()

        for batch in eval_dataloader:
            inputs = torch.tensor(batch[0]).to("cuda")
            with torch.no_grad():
                prob, attention = self.forward(inputs, output_attentions=True)
        return attention
    
    def get_attentions_output(self, dataset, layer, batch_size):
        # dataset = [[torch.tensor(x)] for x in dataset]
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=0,pin_memory=False)

        self.eval()
        attentions = []
        attention_outputs = []
        attention_masks = []
        input_ids = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda").squeeze(1)
            inputs = inputs.view(-1, self.args.block_size)
            inputs = inputs.view(-1, inputs.shape[-1])
            attention_mask=inputs.ne(self.tokenizer.pad_token_id)
            attention_mask = AttentionMaskConverter._expand_mask(mask=attention_mask, dtype=torch.float32, tgt_len=None)
            with torch.no_grad():
                attention = self.encoder.embed_tokens(inputs) * self.encoder.embed_scale
                embed_pos = self.encoder.embed_positions(inputs)
                attention = self.encoder.layernorm_embedding(attention + embed_pos)
                attention = attention, None
                for i in range(layer):
                    attention = self.encoder.layers[i](attention[0], attention_mask=attention_mask, layer_head_mask=None)
                attention_output = self.encoder.layers[i].self_attn(attention[0], attention_mask=attention_mask, layer_head_mask=None)
                attention_outputs.append(attention_output[0])
                attentions.append(attention[0])
                input_ids.append(inputs)

        return torch.cat(attentions, 0), None, torch.cat(input_ids, 0), torch.cat(attention_outputs, 0)

    def continue_forward(self, data, position_bias, input_ids, layer, batch_size):
        self.eval()

        logits = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            inputs = input_ids[i:i+batch_size]
            ori_attention_mask = inputs.ne(self.tokenizer.pad_token_id)
            attention_mask = AttentionMaskConverter._expand_mask(mask=ori_attention_mask, dtype=torch.float32, tgt_len=None)
            attention = batch, None
            self.query += len(batch)

            with torch.no_grad():
                for i in range(layer, 6):
                    attention = self.encoder.layers[i](attention[0], attention_mask=attention_mask, layer_head_mask=None)
                encoder_hidden_state = attention[0]
                
                hidden_states = encoder_hidden_state
                eos_mask = inputs.eq(self.tokenizer.eos_token_id)
                hidden_states = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

                prob = self.classifier(hidden_states)
                prob = F.softmax(prob, dim=-1)
            logits.append(prob.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = []
        for logit in logits:
            pred_labels.append(np.argmax(logit))
        
        return probs, pred_labels
    
    def get_attentions_outputs(self, dataset, batch_size):
        dataset = [[torch.tensor(x)] for x in dataset]
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=0,pin_memory=False)

        self.eval()
        attentions = []
        attention_outputs = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda").squeeze(1)
            inputs = inputs.view(-1, self.args.block_size)
            inputs = inputs.view(-1, inputs.shape[-1])
            attention_mask=inputs.ne(self.tokenizer.pad_token_id)
            attention_mask = AttentionMaskConverter._expand_mask(mask=attention_mask, dtype=torch.float32, tgt_len=None)
            layer_attentions = []
            attention_temp = []
            with torch.no_grad():
                attention = self.encoder.embed_tokens(inputs) * self.encoder.embed_scale
                embed_pos = self.encoder.embed_positions(inputs)
                attention = self.encoder.layernorm_embedding(attention + embed_pos)
                attention = attention, None
                for i in range(6):
                    attention_output = self.encoder.layers[i].self_attn(attention[0], attention_mask=attention_mask, layer_head_mask=None)
                    attention = self.encoder.layers[i](attention[0], attention_mask=attention_mask, layer_head_mask=None)
                    attention_temp.append(attention_output[0])
                    layer_attentions.append(attention[0])
                attention_outputs.append(torch.stack(attention_temp, dim=1).detach().cpu())
                attentions.append(torch.stack(layer_attentions, dim=1).detach().cpu())

        return torch.cat(attentions, 0), torch.cat(attention_outputs, 0)