import torch
import torch.nn as nn
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


class CodeT5RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features.reshape(-1, features.size(-1)  * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = CodeT5RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None, inputs_embeds=None, output_attentions=False):
        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        if inputs_embeds is not None:
            outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                   labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True, output_attentions=output_attentions)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True, output_attentions=output_attentions)
        attentions = outputs.encoder_attentions
        hidden_states = outputs['decoder_hidden_states'][-1]
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
        if output_attentions:
            return prob, attentions
        else:
            return prob

    def get_results(self, dataset, batch_size, threshold=0.5):
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)

        eval_loss = 0.0
        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]

        return probs, pred_labels
    
    def get_attentions_output(self, dataset, layer, batch_size):
        # dataset = [[torch.tensor(x)] for x in dataset]
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=0,pin_memory=False)

        self.eval()
        attentions = []
        attention_masks = []
        position_bias = []
        input_ids = []
        attention_outputs = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda").squeeze(1)
            inputs = inputs.view(-1, self.args.block_size)
            attention_mask=inputs.ne(self.tokenizer.pad_token_id)
            attention_mask = get_extended_attention_mask(attention_mask, inputs.size(), inputs.device)
            with torch.no_grad():
                attention = self.encoder.shared(inputs)
                attention = attention, None
                for i in range(layer):
                    attention = self.encoder.encoder.block[i](attention[0], attention_mask = attention_mask, position_bias = attention[1])
                attention_output = self.encoder.encoder.block[layer].layer[0].SelfAttention(attention[0], mask = attention_mask, position_bias = attention[1])
                attention_outputs.append(attention_output[0])
                attentions.append(attention[0])
                position_bias.append(attention[1])
                attention_masks.append(attention_mask)
                input_ids.append(inputs)

        return torch.cat(attentions, 0), torch.cat(position_bias, 0), torch.cat(input_ids, 0), torch.cat(attention_outputs, 0)

    def continue_forward(self, data, position_bias, input_ids, layer, batch_size):
        self.eval()

        logits = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            inputs = input_ids[i:i+batch_size]
            ori_attention_mask = inputs.ne(self.tokenizer.pad_token_id)
            attention_mask = get_extended_attention_mask(ori_attention_mask, inputs.size(), inputs.device)
            bias = position_bias[i:i+batch_size]
            attention = batch, bias
            self.query += len(batch) / 2

            with torch.no_grad():
                for i in range(layer, 12):
                    attention = self.encoder.encoder.block[i](attention[0], attention_mask = attention_mask, position_bias = attention[1])
                encoder_hidden_state = self.encoder.encoder.final_layer_norm(attention[0])
                decoder_input_ids = self.encoder.decoder._shift_right(inputs)
                decoder_hidden_state = self.encoder.decoder(input_ids=decoder_input_ids, encoder_hidden_states = encoder_hidden_state, encoder_attention_mask=ori_attention_mask)
                
                hidden_states = decoder_hidden_state["last_hidden_state"]
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
