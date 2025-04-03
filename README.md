# v-case-study
A small experiment for an ASR company case study.

## Problem
Given a list of named entities and a speech signal, use this list of possible entities to improve the quality of ASR transcription. To demonstrate approaches, only a conceptual implementation is required.

## Approaches
We propose two approaches to this problem to replace the current one:

### Complete end-to-end approach
Add a special token \<ENTITY> to the vocabulary of the ASR model. During decoding, if the model emits this token, then it is constrained to emit one of the entities in the provided entity list after, followed by another \<ENTITY>.

### Cascaded approach
Use another model (can be a bi-directional LM-like model, or some model also taking) to detect the possible entities in the decoded transcription, and then gradually replace the detected entities with actual entities from the provided entity list in a manner similar to a label-synchronous beam search

## Implementation
The code implements a toy models (RNNT, external LM, greedy decoder, entity recognizer, etc.) together with functionalities for the approaches described above. To make it feasible to implement in a short amount of time, this implementation has lots of caveats, such as the decoder actually uses the RNA topology (while the toy model is trained with standard RNNT loss with vertical transition), entities are restricted to be just one token, etc.

To run tests on dummy inputs, simply run the script `test_dummy_input.py`.
