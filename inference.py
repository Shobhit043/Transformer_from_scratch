
'''
Run this after Training the model

During Trarining the model weights would automatically get stored in the weights/tmodel_{epoch}.pt along with the tokenizers for
the source and target languages

'''
import os
import torch
import torch.nn as nn
from train import get_model, greedy_decode
from tokenizers import Tokenizer
from config import get_config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_encoder_input_and_mask(src_sentence: str, tokenizer_src, seq_len):
    sos_token = torch.tensor(tokenizer_src.token_to_id('[SOS]'), dtype=torch.int64)
    eos_token = torch.tensor(tokenizer_src.token_to_id('[EOS]'), dtype=torch.int64)
    pad_token = torch.tensor(tokenizer_src.token_to_id('[PAD]'), dtype=torch.int64)

    encoder_tokens = tokenizer_src.encode(src_sentence).ids
    enc_pad_len = seq_len - len(encoder_tokens)  - 2
    
    if enc_pad_len<0:
      raise ValueError("Sentence length is greater than Sequence length")
    
    encoder_padded_input_tokens = torch.cat([
        sos_token.reshape(1),
        torch.tensor(encoder_tokens, dtype=torch.int64),
        eos_token.reshape(1),
        torch.tensor (enc_pad_len*[pad_token], dtype = torch.int64)
    ])

    return {
       'encoder_input': encoder_padded_input_tokens.unsqueeze(0),
       'encoder_mask': (encoder_padded_input_tokens != pad_token).unsqueeze(0).unsqueeze(0).int()
    }

def make_prediction(tmodel, src_sentence, tokenizer_src, tokenizer_trg):
    encoder_dict = get_encoder_input_and_mask(src_sentence, tokenizer_src, config['seq_len'])
    model_output = greedy_decode(tmodel, encoder_dict['encoder_input'], encoder_dict['encoder_mask'], tokenizer_src, tokenizer_trg, config['seq_len'], device) 
    prediction = tokenizer_trg.decode(model_output.detach().cpu().numpy())
    return prediction

# get tokenizers
config = get_config()
tokenizer_src = Tokenizer.from_file(str(f"tokenizer_{config['src_lang']}.json"))
tokenizer_trg = Tokenizer.from_file(str(f"tokenizer_{config['trg_lang']}.json"))

# get model and optimizer
tmodel = get_model(config,tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size())
tmodel = tmodel.to(device)
optimizer = torch.optim.Adam(tmodel.parameters(), lr = config['lr'], eps = 1e-9)

# loading model state
if(len(os.listdir('weights')) == 0):
   raise "No model weights found"

model_filepath = os.path.join('weights', os.listdir('weights')[-1]) # getting the latest trained model weights
state = torch.load(model_filepath, map_location=device)
tmodel.load_state_dict(state['model_state_dict'])


src_sentence = "Hello everyone"
predicted_sentence = make_prediction(tmodel, src_sentence, tokenizer_src, tokenizer_trg)

print("\nModel Prediction = " , predicted_sentence,"\n")