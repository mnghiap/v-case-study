"""
Perform some tests on dummy input
"""

import random
import multiprocessing as mp

import torch
import torchaudio

from model.asr.rnnt.rnnt import RNNT
from model.asr.rnnt.rnnt_decoder import RNNT_Decoder
from model.lm.lstm_lm import LSTM_LM
from model.vocab import Vocab

# set num threads to max threads
torch.set_num_threads(mp.cpu_count())

# reproducibility, since there's a lot of random stuffs
random.seed(1337)
torch.manual_seed(1337)

# construct the vocab
vocab_txt = "dummy_data/vocab.txt"
entity_list_txt = "dummy_data/entity_list.txt"
vocab = Vocab(
    vocab_txt,
    entity_list_txt,
    sb_symbol="<SB>",
    blank_symbol="<BLANK>",
    entity_boundary_symbol="<ENTITY>",
    )
# print(vocab.generate_entity_mask())

# some constants value
AUDIO_SAMPLING_RATE = 16000
N_AUDIO_FEATURES = 10
VOCAB_SIZE_NO_BLANK = vocab.vocab_size_no_blank

# construct configs for the RNNT model
feature_extraction_config = dict(
    sample_rate=AUDIO_SAMPLING_RATE,
    n_mfcc=N_AUDIO_FEATURES,
    dct_type=2,
    norm="ortho",
    log_mels=True,
    melkwargs=dict(
        n_fft=400,
        hop_length=160,
        n_mels=20
    )
)

encoder_config = dict(
    input_dim=N_AUDIO_FEATURES,
    hidden_dim=64,
    output_dim=VOCAB_SIZE_NO_BLANK+1,
    n_lstm_layers=1,
)

predictor_config = dict(
    bos_idx=vocab.sb_idx,
    n_lstm_layers=1,
    input_dim=VOCAB_SIZE_NO_BLANK,
    embed_dim=32,
    hidden_dim=64,
    output_dim=VOCAB_SIZE_NO_BLANK+1,
)

joiner_config = dict()

rnnt_config = dict(
    feature_extraction_config=feature_extraction_config,
    encoder_config=encoder_config,
    predictor_config=predictor_config,
    joiner_config=joiner_config
)

# construct external LM config
lm_config = dict(
    bos_idx=vocab.sb_idx,
    eos_idx=vocab.sb_idx,
    n_lstm_layers=1,
    input_dim=VOCAB_SIZE_NO_BLANK,
    embed_dim=32,
    hidden_dim=64,
    output_dim=VOCAB_SIZE_NO_BLANK,
)

# construct actual models and the decoder
rnnt = RNNT(**rnnt_config)
lm = LSTM_LM(**lm_config)

# construct dummy input
raw_wave = torch.rand(1, 16000) * 2 - 1
targets = torch.tensor([[6,2,6,6,5,6,6,3,6]]).int()
batch_size = raw_wave.shape[0]

# and let the RNNT it "learn" one single dummy target
rnnt_optim = torch.optim.AdamW(params=rnnt.parameters(), lr=1e-3)
print("Train the dummy RNNT, printing loss every 100 epochs")
for ep in range(2500):
    logits = rnnt(raw_wave, targets)
    logits_lengths = torch.full((batch_size,), fill_value=logits.shape[1], dtype=torch.int)
    target_lengths = torch.full((batch_size,), fill_value=targets.shape[1], dtype=torch.int)
    loss = torchaudio.functional.rnnt_loss(
        logits,
        targets,
        logit_lengths=logits_lengths,
        target_lengths=target_lengths,
        blank=vocab.blank_idx,
        reduction="mean"
    )
    loss.backward()
    rnnt_optim.step()
    if ep % 100 == 0:
        print(loss)
print()
# now let the LM learn the same sequence as well
lm_optim = torch.optim.AdamW(params=lm.parameters(), lr=1e-3)
print("Train the dummy external LM")
for ep in range(1000):
    ce = -lm.compute_lm_score(targets) # notice the minus sign, because this function computes log p(seq)
    ce.backward()
    lm_optim.step()
    if ep % 100 == 0:
        print(ce)

# construct the decoder to do decoding
decoder = RNNT_Decoder(rnnt, lm, vocab)
hyps = decoder.greedy_search(raw_wave, lm_scale=0.00)
print(hyps)
hyps = decoder.greedy_search(raw_wave, lm_scale=1.50)
print(hyps)
hyps = decoder.greedy_search(raw_wave, lm_scale=0.00, force_entity_constraint=True)
print(hyps)
hyps = decoder.greedy_search(raw_wave, lm_scale=1.50, force_entity_constraint=True)
print(hyps)

