# SCRIBE Backlog

Known weaknesses in transcription quality, ranked by impact on real-world recordings.
Evidence comes from the `input/poor audio test/` sample runs (5-min clips of a meeting,
a 1:1 interview, and a video shoot interview).

---

## Findings from 2026-06-12 restoration experiments

Two compute-budgeted test batteries (`scripts/test_battery.py`,
`scripts/test_battery_v2.py`) gave us hard data:

- **All three sample files were 15–18 dB under broadcast loudness.** Simple
  ffmpeg `loudnorm` + `dynaudnorm` ("meeting" preset) gave the biggest single
  win we've seen: language detection no longer flipped to Norwegian, the
  videointerview's quieter interviewer was loud enough for diarization to
  split SPEAKER_00 + SPEAKER_01, and Whisper recovered content from a segment
  that was previously reduced to "Ja, det tror jeg det er bra."
- **Lowering VAD onset/offset (0.50/0.36 → 0.20/0.15) doubled segment count
  on conference (3 → 7) — but every new segment was a Whisper hallucination**
  ("Danske tekster af Nicolai Winther", "Tak fordi du så med", or a
  "Det kan jeg ikke gøre."×4 loop). VAD threshold is *not* the bottleneck.
- **`afftdn` (FFT denoise) is net-negative on speech** — it triggered the
  repetition loop above. Removed from the `noisy` preset.
- **WPE single-channel dereverb is neutral.** Made one segment worse, added
  one hallucination, no net gain. WPE shines on mic arrays, not mono.
- **DeepFilterNet v0.5.6 made things worse.** Three settings tested
  (full/light/post-filter). At full attenuation 54–77 % of audio got
  classified as silence — DFN treats distant speakers as noise and
  suppresses them. Confirmed via prebuilt aarch64 binary, no Rust toolchain
  required. *Dropped from pipeline; binary archived under `.tools/` for
  future close-mic experiments.*
- **Hard ceiling reached on signal-restoration approaches.** Our recording
  is far-field multi-speaker — fundamentally a different problem class than
  "noisy close-mic" that DFN/WPE target. Next steps need a paradigm shift:
  **speech separation** (not enhancement) or **diarization-first** stacks.

## Calibrating expectations: where SoTA actually lands

CHiME-8 NOTSOFAR-1 single-channel evaluation (2024) is our exact problem
class. The leaderboard tells us the achievable floor:

| Rank | Team | System | Eval tcpWER |
|---|---|---|---|
| 1 | USTC-NERCSLIP | DCF-DS (joint diarization+separation + WavLM-Whisper) | **22.2 %** |
| 2 | NPU-TEA | sys4 | 30.0 % |
| 3 | NJU-AALab | sys1 | 33.5 % |
| 7 | NOTSOFAR baseline | CSS + Whisper + NeMo | 41.4 % |

The best lab in the world lands at 22 % on this condition. "Perfect
transcript" is not on the menu — calibrate user expectations accordingly.

**Architectural lesson, consistent across all top systems:** diarization-aware
**speech separation** (trained on realistic data) before ASR. Not denoise,
not dereverb. Our test batteries confirmed this from the failure side.

## Candidate tools (license-friendly, installable on macOS arm64)

Surveyed 2026-06-12 against our constraints (Apache/MIT/BSD or commercial-OK,
no CUDA hard requirement, installable without compiling Rust/CUDA kernels).

| Tool | License | macOS-CPU? | Trained on | Verdict |
|---|---|---|---|---|
| **pyannote `speech-separation-ami-1.0` (PixIT)** | MIT (code), gated open weights | yes | **AMI-SDM real far-field meetings** | **TOP PICK** — measured 38.8→32.8 % cpWER on AMI-SDM with the full leakage-removal pipeline. Joint diarization+separation. Already in our dep tree. |
| **hviske-v2 (Danish-Whisper, faster-whisper conversion)** | OpenRAIL (commercial OK) | yes | CoRal + Common Voice Danish | **TOP PICK** — 11.8 % WER on Danish; one-line model-string swap in WhisperX. Strong Danish LM prior should kill our `"Danske tekster af X"` hallucinations. |
| **NOTSOFAR-1 baseline CSS (Conformer)** | MIT code, CC-BY 4.0 weights | needs porting (Linux-targeted) | NOTSOFAR simulated 1000 h | Fallback if PixIT underdelivers. Heavier integration; trained on simulated mixtures (more brittle than AMI-real). |
| **MahmoudAshraf97 `whisper-diarization`** | BSD-2 | yes (Apple-Silicon patches landed) | n/a (orchestration) | Reference implementation of faster-whisper + NeMo MSDD + forced alignment. Use as a comparison harness, not a dep. |
| **NeMo MSDD diarizer** | Apache 2.0 | painful (`triton` is Linux-only) | English + simulated | Conditional. Fixes attribution, **not** the deletions driving our WER. Defer until separation is in place. |
| **NeMo Sortformer** | **CC-BY-NC-4.0 — DEALBREAKER** | n/a | n/a | Cannot use commercially. Drop from list. |
| **SpeechBrain `sepformer-whamr16k`** | Apache 2.0 | yes | WHAMR (synthetic WSJ0 2-spk fully-overlapped) | **TRAP — do not use.** SpeechBrain explicitly disclaims out-of-domain performance. On long-form real meetings these models over-separate, smear single-speaker audio across channels, and produce the same artifact class that made DFN hallucinate. |
| **Asteroid ConvTasNet / DPRNN** | MIT | yes | WSJ0 / WHAM (synthetic) | Same trap as SepFormer-WHAMR. Most checkpoints are 8 kHz (throws away half of Whisper's input bandwidth). Drop. |
| **Resemble Enhance** | MIT | slow on CPU | denoise+dereverb | Same family as DFN. Drop. |

## Ranked test-battery proposal (next sprint)

Each item is a single falsifiable experiment that fits one
compute-budgeted run.

### B1 — Free decoder hardening (no model change)
**Status: DONE (2026-06-12), smoke-tested on real audio (2026-06-13).**
Implemented as `HARDENED_ASR_OPTIONS` in
[src/scribe/transcribe.py](src/scribe/transcribe.py) — defaults for every
run now disable `condition_on_previous_text`, tighten the compression
ratio threshold to 2.0, drop near-silence at `no_speech_threshold=0.5`,
and pass `hallucination_silence_threshold=0.5` to faster-whisper (with a
graceful fall-back for older builds that don't recognise the latter).
Smoke test on `conference_5min.m4a` produced bit-identical output to the
pre-hardening "meeting" preset run from 2026-06-12 — same 3 segments,
same text, no regressions. This particular clip never triggered the
pathologies B1 attacks (no `"Ja. Ja. Ja."` loops or YouTube-caption
hallucinations in the prior output), so the change is silent here.
Quantitative evidence on a clip that *does* trigger the loops requires
B4's hand-curated reference.

**Hypothesis:** Our repetition-loop hallucinations (`"Ja. Ja. Ja."`,
`"Tak fordi du så med"`) are partly Whisper-default-decoder pathology, not
inevitable. faster-whisper/WhisperX exposes four knobs that specifically
attack this:
- `condition_on_previous_text=False` — breaks the self-reinforcing loop
- `compression_ratio_threshold≈2.0` — drops segments whose token entropy
  collapses (the signature of a loop)
- `no_speech_threshold≈0.5` — discards near-silent segments before they
  hallucinate subtitle credits
- `hallucination_silence_threshold` (newer WhisperX) — explicit silence
  detector to kill known-bad outputs

**Effort:** ~10 lines in [src/scribe/transcribe.py](src/scribe/transcribe.py).
~5 min compute to re-run all 3 test files.
**Risk:** Slightly more conservative segmentation may drop a real edge
segment. Mitigation: A/B against current output.
**Why first:** zero new dependencies, zero new models, attacks the most
visible failure mode in our existing outputs.

### B2 — Swap to hviske-v2 (Danish-Whisper) via CTranslate2
**Status: DONE — wiring + docs + initial A/B (2026-06-13). Result:
hviske-v2 is NOT a win on far-field meeting audio.** The CLI already
accepted `-m/--model`, so the only work was verifying both candidate
model strings (`pluttodk/hviske-tiske`,
`syvai/faster-hviske-v3-conversation`) load cleanly via faster-whisper
on macOS-arm64-CPU. Both load fine; see [docs/USAGE.md](docs/USAGE.md#use-a-danish-fine-tuned-whisper-recommended-for-danish-audio)
for the user-facing instructions.

**Direct A/B on a 5-minute far-field Danish meeting clip (3 speakers):**

| | large-v3 + B1 hardening | hviske-tiske |
|---|---|---|
| Wall clock | 8m30s | **4m20s** (2× faster) |
| Capitalisation/punctuation | yes | none |
| Sample seg 1 (mild distortion) | recognisable Danish phrase | same phrase mis-spelled (two clear substitution errors) |
| Sample seg 2 (noisy middle) | mostly correct with one minor sub | wholesale hallucination, no resemblance to source |
| Sample seg 3 | Danish | **flipped to English** despite `-l da` |

(Verbatim quotes withheld — the source recording is private.
The exact `output/*.md` files are gitignored; see `references/`
for the curation workflow.)

hviske-v2 is trained on CoRal/Common Voice (close-mic read and conversational
speech) and breaks down on far-field meetings: more spelling errors, full
hallucinations on the noisier middle segment, and a Danish→English language
flip that suggests the Danish prefix isn't being forced on the CTranslate2
decoder the way large-v3 forces it. **Recommendation:** do NOT default to
hviske-v2 for our test material. Keep it as an option for close-mic Danish
interviews where the input is clean (we have not tested that condition yet).

Original hypothesis below is preserved for context — it was correct about
the Danish LM prior killing English-subtitle hallucinations on clean audio,
but did not account for hviske-v2's narrow training distribution.

**Hypothesis:** Whisper-large-v3 fights two wars on our recordings: bad
acoustics *and* mediocre Danish. A Danish-fine-tuned Whisper with a strong
Danish LM prior should both reduce substitutions on speech it already
decodes and refuse the English/Norwegian/subtitle-credit hallucinations
(low probability under the Danish LM).
**Effort:** Two model strings to try: `pluttodk/hviske-tiske` or
`syvai/faster-hviske-v3-conversation`. Both load directly via
faster-whisper. ~5 lines in [src/scribe/transcribe.py](src/scribe/transcribe.py)
to make the model configurable; default stays large-v3.
**License gate:** Pin to **hviske-v2 / OpenRAIL** family — hviske-v5.x is
CC-BY-NC-4.0 (dealbreaker). Document explicitly in code comment + README.
**Risk:** Trained on close-mic read/conversational speech, not far-field.
Won't fix overlap deletions. Expected gain: 10–30 % relative on what
Whisper already attempts.
**Effort:** Half a day including A/B run + decision doc.

### B3 — PixIT separation as pre-Whisper stage
**Status: IMPLEMENTED end-to-end on the 5-min far-field clip
(2026-06-13). Phantom-stream defense added: PixIT still produces a
phantom 3rd stream, but the subtitle-credit hallucinations it would emit
are now filtered out at merge time, so the rendered transcript is
clean.** New module [src/scribe/separate.py](src/scribe/separate.py)
wraps `pyannote/speech-separation-ami-1.0` with manual leakage-removal
(we don't trust undocumented pipeline attributes for that step — we apply
the mask ourselves from `diarization.itertracks()` ± a 0.5 s collar).
New flag `--separator pixit` / `-s pixit` on both `scribe transcribe` and
`scribe batch` switches the pipeline to: ffmpeg preprocess → PixIT →
per-stream Whisper (with `diarize=False`) → per-segment hallucination
filter → merge segments sorted by start time, with each segment tagged by
the PixIT speaker label.

**Two-pronged phantom-stream defense (`separate.py`):**
- **Duration filter** — streams whose total active duration is below
  `MIN_STREAM_ACTIVE_SECONDS = 1.5` are dropped before ASR.
- **Energy filter** — `MIN_STREAM_RMS_DBFS = -42.0` dBFS on the active
  samples after leakage masking. Real speakers on our 5-min clip land at
  −30 to −22 dBFS; the phantom stream sits below this in principle but
  in practice can still squeak through after masking (it did on the
  re-run), so this is a belt-and-braces filter.
- **Hallucination regex** (`is_subtitle_hallucination` in `separate.py`,
  applied in `pipeline.py` after the per-stream Whisper pass) — drops
  segments matching the canonical Whisper subtitle-credit attractors
  (`Danske tekster`, `Tekstet af`, `Subtitles by`, `Scandinavian Text
  Service`, `Amara.org`, …, case-insensitive). These strings come from
  YouTube/Netflix subtitle training data and never appear in real
  recordings.

**Direct comparison on the same 5-min far-field Danish meeting clip:**

| | B1 large-v3 | B3 PixIT, pre-filter | B3 PixIT + filters |
|---|---|---|---|
| Wall clock | 8m30s | 11m44s | 9m13s (warm cache) |
| Streams | n/a | 3 | 3 (phantom passed energy gate) |
| Segments emitted | 3 | 7 | 5 |
| Subtitle-credit hallucinations | none observed | 2 (whole stream 2) | **0 (dropped at merge)** |
| Real Danish content | yes | yes on streams 0+1 | yes, all surviving segments |

The phantom 3rd stream still passes the duration *and* energy gates on
this clip (the diarizer's mask is wide enough that the post-mask RMS
stays above −42 dBFS), so the regex post-filter is currently doing the
real work. Output is clean.

**First-time setup (working install procedure):**
1. Accept terms on BOTH:
   - <https://huggingface.co/pyannote/speech-separation-ami-1.0> (the pipeline)
   - <https://huggingface.co/pyannote/separation-ami-1.0> (the underlying separator)
2. `uv pip install -e ".[separate]"` — pulls asteroid + speechbrain.
   `setuptools<81` is also required (asteroid drags torchmetrics 0.11
   which uses removed `pkg_resources`).
3. Note: the audio is preloaded as `{"waveform", "sample_rate"}` to
   work around a broken torchcodec install on macOS (the dylib RPATHs
   to ffmpeg are wrong on Homebrew). We use torchaudio for loading
   instead — see `_load_audio_for_pipeline()` in separate.py.

**Next concrete steps (open questions):**
- Tighten `MIN_STREAM_RMS_DBFS` once we have more clips; current −42 dBFS
  was set with a 3 dB margin above the measured phantom level on the
  5-min clip but the post-mask measurement on the re-run did not hit it.
  May need to measure on a window centred on diarized regions rather
  than the whole active mask.
- Hand-curate a reference for the 5-min clip and score B1 vs B3 with
  meeteval tcpWER for the first objective number.
- Extend the regex tuple if new subtitle-credit attractors show up on
  other clips (the list is in `separate.py:HALLUCINATION_REGEXES`).

**Hypothesis:** `pyannote/speech-separation-ami-1.0` is the only
pretrained separator on the table that was trained on **real reverberant
distant-mic audio** (AMI-SDM). Published gain on its training distribution
is 38.8→32.8 % cpWER with the full leakage-removal pipeline; insertions
dropped 3.4→1.2 (i.e., it specifically suppresses the hallucination
pattern we see). Output is up-to-3 separated speaker streams + diarization
in a single forward pass — feeds straight into WhisperX per stream.
**Effort:** Most expensive item. ~150 lines: gated HF accept, pipeline
load, per-stream loop, leakage-removal logic, output merge with timestamps.
Plus integration into `pipeline.py` as an optional `--separator pixit`
flag. Budget a weekend.
**Risk:** Trained on English (separation is largely
language-agnostic — needs validation). Max 3 concurrent speakers. Slow on
CPU — run conference_5min overnight on M1.
**Hard rule:** Implement the **leakage-removal** step. Their paper shows
without it, separation makes WER *worse* (50.1 % cpWER, +29 % rel.). This
is the same failure mode our DFN/SepFormer-WHAMR experiments would have
hit. Use the full `Pipeline.from_pretrained` object, not the raw separator.

### B4 — Objective evaluation: hand-corrected reference + meeteval tcpWER
**Status: DONE — infrastructure (2026-06-12), references still to be
hand-curated.** New optional dependency group
(`pyproject.toml [project.optional-dependencies] eval = ["meeteval>=0.4.0"]`),
new module [src/scribe/eval.py](src/scribe/eval.py) with a Markdown→STM
parser and a `score_tcpwer()` wrapper, and two new CLI subcommands:
`scribe eval bootstrap -i output/foo.md -o references/foo.ref.stm`
(seeds an editable reference) and
`scribe eval score -r references/foo.ref.stm -i output/foo.md` (reports
tcpWER + missed/false-alarm speakers). Verified end-to-end: identical
ref/hyp = 0.00 % tcpWER on 3978 words; dropping 5 segments from the
hypothesis correctly produced 10.41 % with 414 deletions. Outstanding
human work: hand-correct a 10–15 min reference for at least one
recording. See [docs/USAGE.md](docs/USAGE.md#score-a-transcript-against-a-manual-reference-scribe-eval).

**Hypothesis:** We have no ground truth. Every "this seems better" claim
above is subjective. The cheapest evaluation asset is a
**10–15 minute manually corrected reference transcript of our own
recording** — scored against everything with tcpWER via the MIT-licensed
`meeteval` package. Public AMI-SDM / NOTSOFAR dev sets are valuable but
non-Danish; for Danish far-field, no public benchmark exists, so we make
our own.
**Effort:** ~2 hours of human correction on `conference_5min.m4a` →
reference STM. Then `pip install meeteval`, ~30 lines to convert our
output JSON → STM. ~1 day total.
**Risk:** None. Pure measurement infrastructure. Unblocks all subsequent
A/B claims.
**Why early:** B1, B2, B3 are all much more meaningful once we have a
number. Run B4 in parallel with B1.

### B5 — NeMo MSDD as diarizer (conditional, only if B3 done)
**Hypothesis:** Once separation is solid, the next bottleneck shifts to
speaker attribution across streams. NeMo MSDD is reportedly stronger than
pyannote on messy meeting audio (mixed evidence — MOSLA project found
opposite). A/B-able.
**Effort:** Hard. NeMo's macOS install path breaks on `triton` (Linux-only).
Realistic route: vendor in MahmoudAshraf97's `whisper-diarization`
orchestration, which already has Apple-Silicon CPU fixes. Budget 1–2 days.
**Risk:** Better diarization fixes *attribution*, not *deletions*. Won't
recover speech the ASR never decoded. Defer until B3 is in.

### B6 — Recommended-mic guide for users
**Hypothesis:** Cheapest fix for far-field is *don't record far-field*.
A short doc explaining mic placement, levels, and "if you must record on
a laptop, here's what tcpWER number to expect" sets honest expectations
and saves users from blaming the tool.
**Effort:** ~1 page of Markdown in [docs/USAGE.md](docs/USAGE.md).
**Why after B4:** the tcpWER numbers we collect become concrete
guidance ("close-mic 7 %, lavalier 12 %, laptop at 2 m: 35 %+").

---

## Recommended order

1. **B1** (decoder hardening) — afternoon, zero new deps, hits visible
   failure mode now.
2. **B4** (hand-corrected reference + meeteval) — 1 day, run in parallel
   with B1. Unblocks objective evaluation for everything that follows.
3. **B2** (hviske-v2 Danish-Whisper) — half a day, one model-string
   change. Orthogonal to acoustic improvements.
4. **B3** (PixIT separation) — weekend. The architectural fix. Requires
   B4 to evaluate honestly.
5. **B5** (NeMo MSDD diarizer) — only if B3 plateaus. 1–2 days.
6. **B6** (mic guide) — once we have real tcpWER numbers per condition.

Realistic combined expectation against current baseline if AMI-SDM
improvements transfer: **20–35 % relative tcpWER reduction**. The world's
best systems still sit at 22 % absolute on this condition — we will not
beat them on a CPU.

## Things we explicitly do NOT plan to try

- **SpeechBrain SepFormer-WHAMR / Asteroid ConvTasNet pretrained checkpoints** —
  trained on synthetic 2-speaker fully-overlapped WSJ0 mixtures, will
  smear single-speaker audio and trigger hallucinations on our material.
  The DFN failure mode generalizes.
- **NVIDIA NeMo Sortformer** — CC-BY-NC-4.0, commercial use prohibited.
- **NVIDIA NeMo Canary / Parakeet ASR** — CUDA-strongly-recommended,
  install path is fragile on macOS, and we have no evidence Whisper is
  the bottleneck (B1+B2 should establish that).
- **Training our own CSS / fine-tuning Whisper** — 1000-hour simulated
  set + GPU days. Out of scope unless we end up with a Danish meeting
  corpus.
- **More ffmpeg filter combos** — ceiling reached on the meeting preset.
- **DeepFilterNet variants** — failure mode is architectural.
- **Multi-channel beamforming (WPE, GSS)** — needs stereo/array input we
  don't have.

## 1. Voice Activity Detection drops far-field speech

**Symptom.** On `conference_5min.m4a` (table mic, multiple speakers around a room) only
**3 segments were emitted in 5 minutes of continuous talking**. Long stretches of audible
speech are silently discarded before they reach Whisper.

**Cause.** Pyannote VAD's default `onset`/`offset` thresholds are tuned for close-mic'd
audio. Far-field speech sits below the energy threshold and is classified as silence.

**Fix ideas.**
- Expose `--vad-onset` / `--vad-offset` (or a single `--vad-sensitive` preset) on the
  CLI, plumbed into `transcribe.py` via WhisperX's `vad_options`.
- Optionally pre-amplify / loudness-normalize with ffmpeg (`loudnorm`) before VAD.
- Document a "noisy meeting" preset in `docs/USAGE.md`.

---

## 2. Language auto-detection is fooled by the first 30 seconds

**Symptom.** `conference_5min.m4a` is Danish with English code-switching, but was
detected as Norwegian Nynorsk (`nn`, 0.93). Result: Danish words came out spelled
Norwegian ("noen greit", "være nød", "lige lave").

**Cause.** WhisperX detects language from the first 30 s only. If the opener is
mumbled, far-field, or in a different language than the body, the entire transcript is
decoded with the wrong tokenizer.

**Fix ideas.**
- Sample multiple windows (e.g. 3 × 30 s spread across the file) and vote.
- Add a config default `default_language` (per-user fallback when confidence < 0.8).
- Print the detection confidence and warn loudly when it is low.

---

## 3. Whisper hallucinates repetitions on silence / fade-outs

**Symptom.** `interview_5min.m4a` ends with `"Ja. Ja. Ja. Ja. Ja."` — a classic
Whisper loop emitted when the audio fades into background noise at a segment boundary.

**Cause.** Default decoding has `condition_on_previous_text=True` and a permissive
`no_speech_threshold`, so once a hallucinated token appears it self-reinforces.

**Fix ideas.**
- Set `condition_on_previous_text=False` for `asr_options` (small WER cost, big
  hallucination reduction).
- Raise `no_speech_threshold` (e.g. 0.6) and tighten `log_prob_threshold`.
- Post-process: collapse identical consecutive single-word segments.

---

## 4. Diarization collapses to one speaker when voices are unbalanced

**Symptom.** `videointerview_5min.m4a` has a clear interviewer + interviewee dialogue
but everything is labeled `SPEAKER_00`. The off-camera interviewer is quieter and gets
clustered with the on-camera speaker.

**Cause.** Pyannote's speaker embeddings cluster by voice similarity *and* SNR; a much
quieter second voice often lands in the same cluster as the dominant one.

**Fix ideas.**
- Expose `--min-speakers` / `--max-speakers` on the CLI (pyannote accepts these as
  hard constraints — huge quality jump when you know the answer).
- Consider per-channel diarization when stereo input has a clear L/R split.
- Document the flag prominently for interview workflows.

---

## 5. Non-speech audio (music, SFX, room noise) is transcribed as words

**Symptom.** `videointerview_5min.m4a` opens with `"There are guns looking very clear"`
— that's intro music/ambience being decoded as English speech.

**Cause.** Whisper is trained on captions that sometimes include song lyrics and sound
descriptions; VAD passes any energetic signal through and Whisper happily invents text.

**Fix ideas.**
- Add a stricter `no_speech_threshold` (see #3) — already helps.
- Consider an optional music/speech classifier pre-filter (e.g. `inaSpeechSegmenter`)
  for files known to contain music.
- Trim known-bad intros/outros via a `--skip-head SECONDS` / `--skip-tail SECONDS`
  pair on the CLI for video-shoot material.

---

## Suggested first sprint

If we tackle these in roughly this order the wins compound:

1. **#2 language** — one-line fix, immediately unblocks #1's evaluation.
2. **#3 hallucinations** — flip two decoder flags, low risk, visible quality jump.
3. **#1 VAD sensitivity** — the biggest gain for meetings, needs a real CLI flag.
4. **#4 diarization hints** — small CLI addition, huge for known-format recordings.
5. **#5 music/SFX** — least common; address last with skip-head/tail as a stopgap.
