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
from model.entity_recognizer.blstm_entity_recognizer import BLSTM_Entity_Recognizer
from model.asr.rnnt.entity_rescorer import EntityRescorer

# set num threads to max threads
torch.set_num_threads(mp.cpu_count())

# reproducibility, since there's a lot of random stuffs
random.seed(1337)
torch.manual_seed(1337)

print("Run tests on dummy inputs")
print("This implementation restricts the entities to be just one token")

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
print("Vocabulary:", list(vocab.symbol_to_idx.keys()))
print("List of entity tokens:", vocab.convert_idxs_to_symbols(vocab.entity_idxs))

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
targets = torch.tensor([[6,2,6,6,3,6,6,4,6]]).int()
print("All models will be trained on just this one dummy target sequence")
print("dummy target sequence (idx)", targets.squeeze(0))
print("dummy target sequence (words)", vocab.convert_idxs_to_symbols(targets.squeeze(0)))
print()
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
    ce.sum().backward()
    lm_optim.step()
    if ep % 100 == 0:
        print(ce)
print()

first_approach_str = """
First approach (E2E): directly integrate <ENTITY> token into the ASR model
and optionally, force outputing entities after outputing <ENTITY> token during decoding
construct the decoder to do decoding
"""
print(first_approach_str)
print("Test the RNNT greedy decoder")
print("Reference:", targets, vocab.convert_idxs_to_symbols(targets))
decoder = RNNT_Decoder(rnnt, lm, vocab)
no_lm_hyps = decoder.greedy_search(raw_wave, lm_scale=0.00)
print("no LM decoding", no_lm_hyps, vocab.convert_idxs_to_symbols(no_lm_hyps))
with_lm_hyps = decoder.greedy_search(raw_wave, lm_scale=1.50)
print("with LM decoding", with_lm_hyps, vocab.convert_idxs_to_symbols(with_lm_hyps))
no_lm_constrained_hyps = decoder.greedy_search(raw_wave, lm_scale=0.00, force_entity_constraint=True)
print("no LM, force output entity decoding", no_lm_constrained_hyps, vocab.convert_idxs_to_symbols(no_lm_constrained_hyps))
with_lm_constrained_hyps = decoder.greedy_search(raw_wave, lm_scale=1.50, force_entity_constraint=True)
print("with LM, force output entity decoding", with_lm_constrained_hyps, vocab.convert_idxs_to_symbols(with_lm_constrained_hyps))
print()

second_approach_str = """
Second approach (Cascaded): have an additional model "marking" entity position
and then gradually replace those positions with entity from the list. Similar
to a label-synchronous beam search over all entity positions and hypothesis
rescoring in two-pass decoding.
"""
print(second_approach_str)
entity_recognizer_config=dict(
    n_blstm_layers=1,
    input_dim=vocab.vocab_size_no_blank,
    embed_dim=32,
    hidden_dim=64,
    output_dim=2, # is entity or is not entity
)
entity_recognizer = BLSTM_Entity_Recognizer(**entity_recognizer_config)

# now let the entity recognizer learn where the entities are
entity_recognizer_optim = torch.optim.AdamW(params=entity_recognizer.parameters(), lr=1e-3)
targets_is_entity = torch.tensor([[0,1,0,0,1,0,0,1,0]]).long()
print("Train the dummy entity recognizer")
for ep in range(500):
    ce = entity_recognizer.compute_ce_loss(targets, targets_is_entity)
    loss = ce.sum()
    loss.backward()
    entity_recognizer_optim.step()
    if ep % 100 == 0:
        print(loss)
print()

# construct the rescorer
print("Test the entity rescorer")
rescorer = EntityRescorer(rnnt, lm, vocab, entity_recognizer)
hyp = torch.tensor(with_lm_hyps).long()
print("Sequence to be rescored:", hyp, vocab.convert_idxs_to_symbols(hyp))
hyps_beams = rescorer.rescore_with_allowed_entities(
    raw_wave,
    hyp,
    beam_size=2,
    lm_scale=0.5,
)
print("Top rescored hypotheses:")
print(hyps_beams)
print(vocab.convert_idxs_to_symbols(hyps_beams))
print()

print("Sequence to be rescored:", targets, vocab.convert_idxs_to_symbols(targets))
hyps_beams = rescorer.rescore_with_allowed_entities(
    raw_wave,
    targets.long(),
    beam_size=2,
    lm_scale=0.5,
    rescore_valid_entities=False,
)
print("Top rescored hypotheses:")
print(hyps_beams)
print(vocab.convert_idxs_to_symbols(hyps_beams))
