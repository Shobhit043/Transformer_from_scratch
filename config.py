from pathlib import Path

def get_config():
  return {
      'ds_size':5000,
      'batch_size':32,
      'num_epoch':10,
      'lr':0.01,
      'seq_len':350,
      'd_model':128,
      'src_lang':'en',
      'trg_lang':'it',
      'model_folder':'weights',
      'model_basename':'tmodel_',
      'preload':None,
      'tokenizer_path':'tokenizer_{0}.json',
      'experiment_name': 'runs/tmodel'
  }

def get_weights_file_path(config, epoch: str):
  folder_name = config['model_folder']
  base_name = config['model_basename']
  file_name = f"{base_name}{epoch}.pt"

  return str(Path('.') / folder_name / file_name) # this works because '/' is overloaded operator in pathlib and joins string in path after /