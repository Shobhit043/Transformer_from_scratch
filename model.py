import torch
import torch.nn as nn
import math

# d_model = dimension of embedding vector(512 in the paper)
# vocab_size = number of words in the vocabulary

class InputEmbeddings(nn.Module):
  def __init__(self, d_model: int,vocab_size: int):
    super(InputEmbeddings, self).__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout: float):
    super(PositionalEncoding, self).__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)

    pe = torch.zeros(size = (seq_len, d_model))
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000)/d_model)) # (d_model)

    # shape of pos*div = (seq_len * d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)

    pe = pe.unsqueeze(0) # (1, seq_len, d_model)

    self.register_buffer('pe', pe) # store along with other model parameters but not updated or changed during training process

  def forward(self, x):
    x = x + self.pe[:, :x.shape[1], :]
    return self.dropout(x)

# In the paper this layer is named "Add and Norm"
class LayerNormalization(nn.Module):
  def __init__(self, epsilon: float = 10**-6):
    super(LayerNormalization, self).__init__()
    self.epsilon = epsilon
    self.alpha = nn.Parameter(torch.ones(1)) # Scale parameter
    self.bias = nn.Parameter(torch.zeros(1)) # Shift parameter

  def forward(self, x):
    mean = torch.mean(x, dim = -1, keepdim=True)
    std = torch.std(x, dim = -1, keepdim=True)

    return self.alpha*(x-mean)/torch.sqrt(std + self.epsilon) + self.bias

class FeedForward(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float):
    super(FeedForward, self).__init__()
    self.Dense1 = nn.Linear(d_model,d_ff) # w1 and b1
    self.dropout = nn.Dropout(dropout)
    self.Dense2 = nn.Linear(d_ff, d_model) # w2 and b2

  def forward(self, x):
    # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
    x = self.Dense1(x)  # x*w1 + b1
    x = nn.ReLU()(x)    # max(0, x*w1 + b1)
    x = self.dropout(x) # dropout(max(0, x*w1 + b1))
    x = self.Dense2(x)  # dropout(max(0, x*w1 + b1))*w2 + b2
    return x

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float):
    super(MultiHeadAttention, self).__init__()

    assert (d_model%h == 0),"d_model is not divisible by h"

    self.h = h
    self.d_model = d_model

    self.d_k = d_model//h
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def Attention(q, k, v, mask, dropout):
    d_k = q.shape[-1]

    attention_scores = (( q @ (k.transpose(-2, -1)))/math.sqrt(d_k))

    if mask is not None:
      attention_scores = attention_scores.masked_fill(mask == 0, -1e9) # -1e9 represents negative infinity ( we want to set some key value pairs to be zero(after softmax) to have no effect or contribution)

    attention_scores = attention_scores.softmax(dim = -1)

    if dropout is not None:
      attention_scores = dropout(attention_scores)

    return (attention_scores @ v), attention_scores # attention_scores would be later used for visualization

  def forward(self, q, k, v, mask):
    query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
    key = self.w_k(k) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
    value = self.w_v(v) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)

    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k) # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k) # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k) # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)

    query = query.transpose(1, 2) # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
    key = key.transpose(1, 2) # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
    value = value.transpose(1, 2) # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)

    x, attention_scores = MultiHeadAttention.Attention(query, key ,value , mask, self.dropout)

    x = x.transpose(1, 2).contiguous().view(query.shape[0], -1, self.d_k*self.h) # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)

    return self.w_o(x)


class ResidualConnection(nn.Module):
  def __init__(self, dropout: float):
    super(ResidualConnection, self).__init__()
    self.dropout = nn.Dropout(dropout)
    self.norm = LayerNormalization()
  def forward(self, x, prevlayer):
    return x + self.dropout(prevlayer(self.norm(x))) # in the original paper they have first applied prevlayer and then norm but most of the implementations have done this

class EncoderBlock(nn.Module):
  def __init__(self, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
    super(EncoderBlock, self).__init__()
    self.self_attention_block = self_attention
    self.feed_forward_block = feed_forward
    self.residual_connection = nn.ModuleList([ResidualConnection(dropout), ResidualConnection(dropout)])

  def forward(self, x, src_mask):
    x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # we are passing the prevlayer in form of lambda function to handle MultiHead inputs
    x = self.residual_connection[1](x, self.feed_forward_block)
    return x

class Encoder(nn.Module):
  def __init__(self, layers: nn.ModuleList):
    super(Encoder, self).__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x, mask):
    for layer in self.layers:
      x = layer(x, mask)
    return self.norm(x)

class DecoderBlock(nn.Module):
  def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: float):
    super(DecoderBlock, self).__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block
    self.residual_connection = nn.ModuleList([ResidualConnection(dropout), ResidualConnection(dropout), ResidualConnection(dropout)])

  def forward(self, decoder_input, encoder_output, src_mask, trg_mask):
    decoder_out = self.residual_connection[0](decoder_input, lambda decoder_input: self.self_attention_block(decoder_input, decoder_input, decoder_input, trg_mask))
    decoder_out = self.residual_connection[1](decoder_out, lambda decoder_out: self.cross_attention_block(decoder_out, encoder_output, encoder_output, src_mask))
    decoder_out = self.residual_connection[2](decoder_out, self.feed_forward_block)

    return decoder_out

class Decoder(nn.Module):
  def __init__(self, layers: nn.ModuleList):
    super(Decoder, self).__init__()
    self.layers = layers
    self.norm = LayerNormalization()

  def forward(self, x, encoder_output, src_mask, trg_mask):
    for layer in self.layers:
      x = layer(x, encoder_output, src_mask, trg_mask)

    return self.norm(x)

class ProjectionLayer(nn.Module):
  def __init__(self, d_model: int, vocab_size: int):
    super(ProjectionLayer, self).__init__()
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    return torch.log_softmax(self.proj(x), dim = -1) # instead of softmax we apply log_softmax for numerical stability


class Transformer(nn.Module):
  def __init__(self, encoder: Encoder, decoder: Decoder, src_embed, trg_embed, src_pos, trg_pos, proj_layer):
    super(Transformer, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.trg_embed = trg_embed
    self.src_pos = src_pos
    self.trg_pos = trg_pos
    self.proj_layer = proj_layer

  def encode(self, src, mask):
    src = self.src_embed(src)
    src = self.src_pos(src)
    return self.encoder(src, mask)

  def decode(self, encoder_out, src_mask, trg, trg_mask):
      trg = self.trg_embed(trg)
      trg = self.trg_pos(trg)
      return self.decoder(trg, encoder_out, src_mask, trg_mask)

  def project(self, x):
    return self.proj_layer(x)

def build_transformer(src_seq_len: int, trg_seq_len: int, src_vocab_size: int, trg_vocab_size: int, N: int = 6, h: int = 8,d_model: int = 64, d_ff: int = 2048, dropout: float = 0.1) ->Transformer :
  src_embed = InputEmbeddings(d_model, src_vocab_size)
  trg_embed = InputEmbeddings(d_model, trg_vocab_size)

  src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
  trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

  encoder_blocks = []
  for _ in range(N):
    self_attention_block = MultiHeadAttention(d_model, h, dropout)
    feed_forward_block = FeedForward(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
    encoder_blocks.append(encoder_block)

  decoder_blocks = []
  for _ in range(N):
    self_attention_block = MultiHeadAttention(d_model, h, dropout)
    cross_attention_block = MultiHeadAttention(d_model, h, dropout)
    feed_forward_block = FeedForward(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout)
    decoder_blocks.append(decoder_block)

  encoder = Encoder(nn.ModuleList(encoder_blocks))
  decoder = Decoder(nn.ModuleList(decoder_blocks))

  proj_layer = ProjectionLayer(d_model, trg_vocab_size)

  transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, proj_layer)

  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return transformer