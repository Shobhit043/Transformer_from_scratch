from dataset import LangDataset, diagonal_mask
from model import build_transformer
from config import get_weights_file_path, get_config
import torch
import torch.nn as nn
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

def greedy_decode(tmodel, encoder_input, encoder_mask, tokenizer_src, tokenizer_trg, maxlen, device):
  sos_idx = tokenizer_src.token_to_id('[SOS]')
  eos_idx = tokenizer_src.token_to_id('[EOS]')

  encoder_output = tmodel.encode(encoder_input, encoder_mask)

  decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(encoder_input).to(device)

  while True:
    if decoder_input.shape[1] == maxlen:
      break

    decoder_mask = diagonal_mask(decoder_input.size(1)).type_as(encoder_input).to(device)

    decoder_output = tmodel.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

    prob = tmodel.project(decoder_output[:, -1]) 

    _, next_word = torch.max(prob, dim = 1)

    decoder_input = torch.cat([
      decoder_input,
      torch.empty(1, 1).fill_(next_word.item()).type_as(encoder_input).to(device)
    ], dim = 1)

    if next_word.item() == eos_idx:
      break

  return decoder_input.squeeze(0)

def eval_validation(tmodel, validation_ds, device, tokenizer_src, tokenizer_trg, maxlen, print_msg, num_examples = 1):
  tmodel.eval()
  count = 0

  with torch.no_grad():
    for batch in validation_ds:
      count += 1
      encoder_input = batch['encoder_input'].to(device)
      encoder_mask = batch['encoder_mask'].to(device)

      assert encoder_input.size(0) == 1, "Batch size must be 1 for validation eval"
      model_output = greedy_decode(tmodel, encoder_input, encoder_mask, tokenizer_src, tokenizer_trg, maxlen, device)

      src_txt = batch['src_sentence'][0]
      expected_txt = batch['trg_sentence'][0]
      predicted_txt = tokenizer_trg.decode(model_output.detach().cpu().numpy())

      print_msg("\n\n Testing on Validation Dataset\n")
      print_msg(f"\n Source = {src_txt}")
      print_msg(f"\n Expected = {expected_txt}")
      print_msg(f"\n Prediction = {predicted_txt}")
      print_msg(f"\n Prediction len = {len(predicted_txt)} \n")

      if count == num_examples:break



def get_sentences(ds, lang):
  for row in ds:
    yield row['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
  tokenizer_path = Path(config['tokenizer_path'].format(lang))
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]","[SOS]","[EOS]","[PAD]"], min_frequency = 2)
    tokenizer.train_from_iterator(get_sentences(ds, lang), trainer = trainer)
    tokenizer.save(path = str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  
  
  return tokenizer

def get_ds_and_tokenizer(config):
  ds = load_dataset("Helsinki-NLP/opus_books", f'{config["src_lang"]}-{config["trg_lang"]}', split = f'train[:{config["ds_size"]}]')

  tokenizer_src = get_or_build_tokenizer(config, ds, config['src_lang'])
  tokenizer_trg = get_or_build_tokenizer(config, ds, config['trg_lang'])

  proportions = [0.9, 0.1]
  lengths = [int(p * len(ds)) for p in proportions]
  lengths[-1] = len(ds) - sum(lengths[:-1])
  train_ds, validation_ds = random_split(ds, lengths)

  print(type(tokenizer_src))
  train_ds = LangDataset(train_ds, tokenizer_src, tokenizer_trg, config["src_lang"], config["trg_lang"], config["seq_len"])
  val_ds = LangDataset(validation_ds, tokenizer_src, tokenizer_trg, config["src_lang"], config["trg_lang"], config["seq_len"])

  max_len_src = 0
  max_len_trg = 0

  for row in ds:
    src_seq_len = len(tokenizer_src.encode(row['translation'][config['src_lang']]).ids)
    trg_seq_len = len(tokenizer_trg.encode(row['translation'][config['trg_lang']]).ids)
    max_len_src = max(max_len_src, src_seq_len)
    max_len_trg = max(max_len_trg, trg_seq_len)

  train_dataloader = DataLoader(train_ds, batch_size = config["batch_size"], shuffle = True)
  val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)

  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg


def get_model(config, vocab_src_len, vocab_trg_len):
  transformer = build_transformer(config['seq_len'], config['seq_len'], vocab_src_len, vocab_trg_len, d_model = config['d_model'])
  return transformer

def train_model(config):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device = {device}")

  Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

  train_dataloader, val_dataloader, tokenizer_src, tokenizer_trg = get_ds_and_tokenizer(config)

  tmodel = get_model(config,tokenizer_src.get_vocab_size(), tokenizer_trg.get_vocab_size())
  tmodel = tmodel.to(device)

  optimizer = torch.optim.Adam(tmodel.parameters(), lr = config['lr'], eps = 1e-9)

  initial_epoch = 0
  global_step = 0

  if config['preload']:
    model_filename = get_weights_file_path(config, config['preload'])
    print(f"Loading model {model_filename}")
    state = torch.load(model_filename)
    tmodel.load_state_dict(state['model_state_dict'])
    initial_epoch = state['epoch']+1
    global_step = state['global_step']
    optimizer.load_state_dict(state['optimizer_state_dict'])
  
  loss_fn = nn.CrossEntropyLoss(
      ignore_index=tokenizer_src.token_to_id('[PAD]'),
      label_smoothing=0.1 # reduce model prediction confidence and allocates some probablity from the highest one to other (reduce overfitting)
  ).to(device)

  for epoch in range(initial_epoch, config['num_epoch']):
    torch.cuda.empty_cache()
    batch_iter = tqdm(train_dataloader, desc = f"Processing epoch {epoch:02d}")
    tmodel.train()
    for batch in batch_iter:
      encoder_input = batch['encoder_input'].to(device)  # (8, seq_len)
      decoder_input = batch['decoder_input'].to(device)  # (8, seq_len)
      encoder_mask = batch['encoder_mask'].to(device)    # (8, 1, 1, seq_len)
      decoder_mask = batch['decoder_mask'].to(device)    # (8, 1, seq_len, seq_len)

      encoder_output = tmodel.encode(encoder_input, encoder_mask) # (8, seq_len, d_model)
      decoder_output = tmodel.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (8, seq_len, d_model)
      projection = tmodel.project(decoder_output).to(device) # (8, seq_len, trg_vocab_len)

      label = batch['decoder_output'].to(device)

      loss = loss_fn(projection.view(-1, tokenizer_trg.get_vocab_size()), label.view(-1))
      batch_iter.set_postfix({'loss': loss.item()})

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


      global_step += 1
    eval_validation(tmodel, val_dataloader, device, tokenizer_src, tokenizer_trg, config['seq_len'], lambda msg: batch_iter.write(msg))
    
    model_file_name = get_weights_file_path(config, f'{epoch:02d}')
    torch.save({
      "epoch" : epoch,
      "model_state_dict" : tmodel.state_dict(),
      "optimizer_state_dict" : optimizer.state_dict(),
      "global_step":global_step    
    },model_file_name)

if __name__=='__main__':
  cfg = get_config()
  train_model(dict(cfg))

