"""MindCart: does EEG improve clothing-preference prediction over product data alone?

Pipeline:
  prep_images.py     -> normalise stimuli + compute low-level image features
  record_session.py  -> record Unicorn EEG (LSL) + experiment markers (LSL) to CSV
  experiment.py      -> PsychoPy: fixation -> product (4 s) -> rating 1-5, sends LSL markers
  features.py        -> epoch trials, baseline-correct, extract band-power features
  models.py          -> Model A (metadata) vs B (EEG) vs C (both), CV + permutation test
  simulate.py        -> fake session with known effect, to test everything without hardware
"""
