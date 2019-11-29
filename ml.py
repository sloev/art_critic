import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
import re

logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

args = 1
output_args = [''] * args
text = '[CLS] a rusted fire hydrant sitting in the middle of a {} forest . [SEP]'

for arg in range(args):
    this_args = [''] * args
    this_args[arg] = '[MASK]'
    this_text = text.format(*this_args)
    tokenized_text = tokenizer.tokenize(this_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    masked_index = this_text.index('[MASK]')
    # masked_index = tokenized_text.index('')
    # print(masked_index)
    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

    print(predictions)
    predicted_index = torch.argmax(predictions[0, -1]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    print('prediction:', predicted_token)
    output_args[arg] = predicted_token

print(text.format(*output_args))