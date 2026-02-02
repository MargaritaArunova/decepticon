import math

import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_prob=0.0):
        """
        :param embedding_dim: dimension of en embedding
        :param num_heads: number of heads
        :param dropout_prob: probability for dropout layer
        """
        super().__init__()
        assert embedding_dim % num_heads == 0

        # number of heads and embedding and head dimensions
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # linear layers using in Q, K and V calculation
        self.WQ = nn.Linear(embedding_dim, embedding_dim)
        self.WK = nn.Linear(embedding_dim, embedding_dim)
        self.WV = nn.Linear(embedding_dim, embedding_dim)
        self.WO = nn.Linear(embedding_dim, embedding_dim)

        # dropout layer to reduce overfitting
        self.dropout = nn.Dropout(dropout_prob)

        # coefficient for scaling using in self-attention calculation
        self.norm_factor = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        """
        :param query: input embeddings, (batch_size, length, embedding_dim)
        :param key: input embeddings, (batch_size, length, embedding_dim)
        :param value: input embeddings, (batch_size, length, embedding_dim)
        :param mask: marks the elements we shouldn't pay attention to
        :return: recalculated embeddings
        """
        batch_size = query.shape[0]

        # compute Q, K and V using linear layers, (batch_size, length, embedding_dim)
        Q = self.WQ(query)
        K = self.WK(key)
        V = self.WV(value)

        # split the embedding_dim of the Q, K and V into num_heads
        # (batch_size, length, embedding_dim) -> (batch_size, num_heads, length, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # calculate dot product (energy), (batch_size, num_heads, length, length)
        dot_product = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.norm_factor
        # mask the dot_product, so we do not pay attention over any elements of the sequence we shouldn't
        if mask is not None:
            dot_product = dot_product.masked_fill(mask == 0, -math.inf)

        # calculate self-attention score, (batch_size, num_heads, length, length)
        attention_score = torch.softmax(dot_product, dim=-1)

        # calculate self-attention, (batch_size, length, num_heads, head_dim)
        attention = torch.matmul(self.dropout(attention_score), V).permute(0, 2, 1, 3).contiguous()
        # combine all heads into one of size embedding_dim
        # (batch_size, length, num_heads, head_dim) -> (batch_size, length, embedding_dim)
        attention = attention.view(batch_size, -1, self.embedding_dim)

        # apply linear layer to get the output, (batch_size, length, embedding_dim)
        output = self.WO(attention)

        return output, attention_score


class FeedforwardLayer(nn.Module):
    def __init__(self, embedding_dim, feedforward_dim, dropout_prob=0.0):
        """
        :param embedding_dim: dimension of en embedding
        :param feedforward_dim: inner-layer dimensionality (usually larger than embedding_dim)
        :param dropout_prob: probability for dropout layer
        """
        super().__init__()

        # position wise feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(feedforward_dim, embedding_dim),
        )

    def forward(self, inputs):
        """
        :param inputs: input embeddings, (batch_size, length, embedding_dim)
        :return: embeddings after feedforward operation
        """
        return self.feedforward(inputs)


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers, num_heads, feedforward_dim,
                 dropout_prob=0.0, max_length=200, device='cpu'):
        """
        :param input_dim: dimension of an input
        :param embedding_dim: dimension of an embedding
        :param num_layers: number of encoder layers
        :param num_heads: number of heads in attention layer in each encoder layer
        :param feedforward_dim: dimension in position wise feedforward layer in each encoder layer
        :param dropout_prob: probability for dropout layer
        :param max_length: maximum length of tokens in string
        :param device: cuda or cpu
        """
        super().__init__()
        self.device = device

        # token embedding
        self.token_embedding = nn.Embedding(input_dim, embedding_dim)

        # positional embedding (make it trainable as in BERT)
        self.positional_embedding = nn.Embedding(max_length, embedding_dim)

        # dropout layer to reduce overfitting
        self.dropout = nn.Dropout(dropout_prob)

        # coefficient for scaling token embedding to reduce model variance
        self.norm_factor = math.sqrt(embedding_dim)

        # encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, feedforward_dim, dropout_prob) for _ in range(num_layers)
        ])

    def forward(self, source_tokens, source_mask):
        """
        :param source_tokens: input source tokens, (batch_size, length)
        :param source_mask: marks the elements equal to <pad>, (batch_size, 1, 1, length)
        :return: sequence of context vectors
        """
        batch_size, length = source_tokens.shape[0], source_tokens.shape[1]

        # encode source tokens and add positional encoding
        positions = torch.arange(0, length).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        outputs = self.norm_factor * self.token_embedding(source_tokens) + self.positional_embedding(positions)

        # apply dropout
        outputs = self.dropout(outputs)

        # apply each encoder layer to outputs
        for encoder_layer in self.encoder_layers:
            outputs = encoder_layer(outputs, source_mask)

        return outputs


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, feedforward_dim, dropout_prob=0.0):
        """
        :param embedding_dim: dimension of en embedding
        :param num_heads: number of heads in attention layer
        :param feedforward_dim: dimension in position wise feedforward layer
        :param dropout_prob: probability for dropout layer
        """
        super().__init__()

        # self-attention layer
        self.self_attention = MultiHeadAttentionLayer(embedding_dim, num_heads, dropout_prob)

        # feedforward layer
        self.feedforward = FeedforwardLayer(embedding_dim, feedforward_dim, dropout_prob)

        # normalization layers that applying after self-attention and feedforward layers
        self.layer_norm_sa = nn.LayerNorm(embedding_dim)
        self.layer_norm_ff = nn.LayerNorm(embedding_dim)

        # dropout layer to reduce overfitting
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs, source_mask):
        """
        :param inputs: input embeddings, (batch_size, length, embedding_dim)
        :param source_mask: (batch_size, 1, 1, length)
        :return: sequence of context vectors
        """
        # apply self-attention
        self_attention_output, _ = self.self_attention(inputs, inputs, inputs, source_mask)

        # apply dropout to output, add residual connection and layer norm
        outputs = self.layer_norm_sa(inputs + self.dropout(self_attention_output))

        # apply position wise feedforward
        feedforward_output = self.feedforward(outputs)

        # apply dropout to output, add residual connection and layer norm
        outputs = self.layer_norm_ff(outputs + self.dropout(feedforward_output))

        return outputs


class Decoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers, num_heads, feedforward_dim,
                 dropout_prob=0.0, max_length=200, device='cpu'):
        """
        :param input_dim: dimension of an input
        :param embedding_dim: dimension of an embedding
        :param num_layers: number of decoder layers
        :param num_heads: number of heads in attention layer in each decoder layer
        :param feedforward_dim: dimension in position wise feedforward layer in each decoder layer
        :param dropout_prob: probability for dropout layer
        :param max_length: maximum length of tokens in string
        :param device: cuda or cpu
        """
        super().__init__()
        self.device = device

        # token embedding
        self.token_embedding = nn.Embedding(input_dim, embedding_dim)

        # positional embedding (make it trainable as in BERT)
        self.positional_embedding = nn.Embedding(max_length, embedding_dim)

        # dropout layer to reduce overfitting
        self.dropout = nn.Dropout(dropout_prob)

        # coefficient for scaling token embedding to reduce model variance
        self.norm_factor = math.sqrt(embedding_dim)

        # decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, feedforward_dim, dropout_prob) for _ in range(num_layers)
        ])

        # final linear layer of decoder
        self.output_layer = nn.Linear(embedding_dim, input_dim)

    def forward(self, target_tokens, encoded_source_embed, target_mask, source_mask):
        """
        :param target_tokens: input target tokens, (batch_size, length)
        :param encoded_source_embed: encoded source embeddings taken from Encoder, (batch_size, length, embedding_dim)
        :param target_mask: marks the elements each token is allowed to look at, (batch_size, 1, length, length)
        :param source_mask: marks the elements equal to <pad>, (batch_size, 1, 1, length)
        :return: sequence of context vectors and attention
        """
        batch_size, length = target_tokens.shape[0], target_tokens.shape[1]

        # encode target tokens and add positional encoding
        positions = torch.arange(0, length).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        outputs = self.norm_factor * self.token_embedding(target_tokens) + self.positional_embedding(positions)

        # apply dropout
        outputs = self.dropout(outputs)

        # apply each decoder layer to outputs
        attention = 0
        for decoder_layer in self.decoder_layers:
            outputs, attention = decoder_layer(outputs, encoded_source_embed, target_mask, source_mask)

        # apply final linear layer
        outputs = self.output_layer(outputs)

        return outputs, attention


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, feedforward_dim, dropout_prob=0.0):
        """
        :param embedding_dim: dimension of en embedding
        :param num_heads: number of heads in attention layer
        :param feedforward_dim: dimension in position wise feedforward layer
        :param dropout_prob: probability for dropout layer
        """
        super().__init__()

        # self-attention and encoder-attention layers
        self.self_attention = MultiHeadAttentionLayer(embedding_dim, num_heads, dropout_prob)
        self.encoder_attention = MultiHeadAttentionLayer(embedding_dim, num_heads, dropout_prob)

        # feedforward layer
        self.feedforward = FeedforwardLayer(embedding_dim, feedforward_dim, dropout_prob)

        # normalization layers that applying after self-attention encoder-attention and feedforward layers
        self.layer_norm_sa = nn.LayerNorm(embedding_dim)
        self.layer_norm_ea = nn.LayerNorm(embedding_dim)
        self.layer_norm_ff = nn.LayerNorm(embedding_dim)

        # dropout layer to reduce overfitting
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs, encoded_source_embed, target_mask, source_mask):
        """
        :param inputs: input target embeddings, (batch_size, length, embedding_dim)
        :param encoded_source_embed: encoded source embeddings taken from Encoder, (batch_size, length, embedding_dim)
        :param target_mask: marks the elements each token is allowed to look at, (batch_size, 1, length, length)
        :param source_mask: marks the elements equal to <pad>, (batch_size, 1, 1, length)
        :return: sequence of context vectors and attention
        """

        # apply self-attention
        self_attention_output, _ = self.self_attention(inputs, inputs, inputs, target_mask)

        # apply dropout to output, add residual connection and layer norm
        outputs = self.layer_norm_sa(inputs + self.dropout(self_attention_output))

        # encoder attention
        self_attention_output, attention = self.encoder_attention(
            outputs, encoded_source_embed, encoded_source_embed, source_mask
        )

        # apply dropout to output, add residual connection and layer norm
        outputs = self.layer_norm_ea(outputs + self.dropout(self_attention_output))

        # apply position wise feedforward
        feedforward_output = self.feedforward(outputs)

        # apply dropout to output, add residual connection and layer norm
        outputs = self.layer_norm_ff(outputs + self.dropout(feedforward_output))

        return outputs, attention


class TranslationModel(nn.Module):
    def __init__(self, encoder, decoder, dataset, device='cpu'):
        """
        :param encoder: encoder model
        :param decoder: decoder model
        :param dataset: dataset for translation model
        :param device: cuda or cpu
        """
        super().__init__()
        self.device = device

        # required in inference mode to decode source sentences
        self.dataset = dataset

        # <pad> indices in source and target sequences
        self.pad_id = dataset.pad_id

        # encoder and decoder models
        self.encoder = encoder
        self.decoder = decoder

    def get_source_mask(self, source):
        pad_mask = (source != self.pad_id).unsqueeze(1).unsqueeze(2)
        return pad_mask

    def get_target_mask(self, target):
        length = target.shape[1]
        pad_mask = (target != self.pad_id).unsqueeze(1).unsqueeze(2)
        sub_mask = torch.tril(torch.ones((length, length), device=self.device)).bool()
        return pad_mask & sub_mask

    def forward(self, source, target):
        # get masks
        source_mask = self.get_source_mask(source)
        target_mask = self.get_target_mask(target)
        # get encoded source embedding
        encoded_source_embed = self.encoder(source, source_mask)
        # get output and attention
        output, attention = self.decoder(target, encoded_source_embed, target_mask, source_mask)

        return output, attention

    @torch.inference_mode()
    def inference(self, sentence, max_length=200):
        """
        :param sentence: input sentence to translate
        :param max_length: maximum number of tokens in translated string
        :return: translated sentence and attention
        """
        # switch model to eval mode
        self.eval()

        # encode sentence and create its mask
        source_tokens = torch.tensor(
            [self.dataset.bos_id] + self.dataset.text2ids(sentence, 'source') + [self.dataset.eos_id],
            dtype=torch.int64
        ).unsqueeze(0).to(self.device)
        source_mask = self.get_source_mask(source_tokens)

        # compute encoded source embedding
        encoded_source_embed = self.encoder(source_tokens, source_mask)

        # create target sentence, continue process until reach max_length or get EOS token
        target_tokens_arr = [self.dataset.bos_id]
        while len(target_tokens_arr) < max_length and target_tokens_arr[-1] != self.dataset.eos_id:
            # make target tensor and create its mask
            target_tokens = torch.tensor(target_tokens_arr, dtype=torch.int64).unsqueeze(0).to(self.device)
            target_mask = self.get_target_mask(target_tokens)

            # get output and attention
            output, attention = self.decoder(target_tokens, encoded_source_embed, target_mask, source_mask)

            # predict next token in translated sentence
            new_target_token = output.argmax(2)[:, -1].item()

            # add new token to result token array
            target_tokens_arr.append(new_target_token)

        return self.dataset.ids2text(target_tokens_arr, 'target')
