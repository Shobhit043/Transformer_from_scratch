import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class LangDataset(Dataset):
  
  def __init__(self, ds, tokenizer_src, tokenizer_trg, src_lang, trg_lang, seq_len):
    print(type(tokenizer_src))
    self.ds = ds
    self.tokenizer_src = tokenizer_src
    self.tokenizer_trg = tokenizer_trg
    self.src_lang = src_lang
    self.trg_lang = trg_lang
    self.seq_len = seq_len

    self.sos_token = torch.tensor(tokenizer_src.token_to_id('[SOS]'), dtype=torch.int64)
    self.eos_token = torch.tensor(tokenizer_src.token_to_id('[EOS]'), dtype=torch.int64)
    self.pad_token = torch.tensor(tokenizer_src.token_to_id('[PAD]'), dtype=torch.int64)


  def __len__(self,):
    return len(self.ds)

  def __getitem__(self, index): 
    src_trg_pair = self.ds[index]

    src_sentence = src_trg_pair['translation'][self.src_lang]
    trg_sentence = src_trg_pair['translation'][self.trg_lang]

    encoder_tokens = self.tokenizer_src.encode(src_sentence).ids
    decoder_tokens = self.tokenizer_trg.encode(trg_sentence).ids

    enc_pad_len = self.seq_len - len(encoder_tokens)  - 2 # add both sos and eos
    dec_pad_len = self.seq_len - len(decoder_tokens)  - 1 # add either sos or eos

    if enc_pad_len<0 or dec_pad_len<0:
      raise ValueError("Sentence length is greater than Sequence length")
    
    encoder_padded_input_tokens = torch.cat([
        self.sos_token.reshape(1),
        torch.tensor(encoder_tokens, dtype=torch.int64),
        self.eos_token.reshape(1),
        torch.tensor (enc_pad_len*[self.pad_token], dtype = torch.int64)
    ])

    decoder_padded_input_tokens = torch.cat([
        self.sos_token.reshape(1),
        torch.tensor(decoder_tokens, dtype=torch.int64),
        torch.tensor (dec_pad_len*[self.pad_token], dtype = torch.int64)
    ])

    decoder_padded_output_tokens = torch.cat([
        torch.tensor(decoder_tokens, dtype=torch.int64),
        self.eos_token.reshape(1),
        torch.tensor (dec_pad_len*[self.pad_token], dtype = torch.int64)
    ])


    assert len(encoder_padded_input_tokens) == self.seq_len
    assert len(decoder_padded_input_tokens) == self.seq_len
    assert len(decoder_padded_output_tokens) == self.seq_len

    return {
        'encoder_input' : encoder_padded_input_tokens,
        'decoder_input' : decoder_padded_input_tokens,
        'decoder_output' : decoder_padded_output_tokens,
        'encoder_mask' : (encoder_padded_input_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
        'decoder_mask' : (decoder_padded_input_tokens != self.pad_token).unsqueeze(0).int() & diagonal_mask(decoder_padded_input_tokens.size(0)), # (1, 1, seq_len) & (seq_len, seq_len)
        'src_sentence' : src_sentence,
        'trg_sentence' : trg_sentence,

    }

def diagonal_mask(size):
  mask = torch.tril(torch.ones((1, size, size), dtype=torch.int))
  return mask == 0