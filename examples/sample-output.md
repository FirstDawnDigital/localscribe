# Sample output

This is a real (lightly trimmed) example of what SCRIBE produces — generated end-to-end from a 22-minute English podcast. Speaker labels in this particular sample read as a single speaker because diarization was not enabled for this run; with `--diarize` (the default) you would see `SPEAKER_00`, `SPEAKER_01`, etc.

The actual file in `output/` is longer; this is the first portion for illustration.

---

```markdown
---
source_file: "Why aren't we all getting rich from compound interest_.mp3"
duration: 00:22:55
language: en
transcribed_at: 2026-06-11T15:38:49+00:00
whisper_model: large-v3
summary_model: gemma2:9b
---

# Why aren't we all getting rich from compound interest_

## Summary

This video argues that while compound interest seems like a path to prosperity
for everyone, it actually benefits the wealthy at the expense of the middle
class.

Here's a breakdown of the key points:

* **Compound Interest Doesn't Benefit Everyone:** The speaker claims that in
  finite economies, where resources are limited, compound interest primarily
  benefits the already wealthy. They argue that as the rich grow their wealth
  faster than the economy itself, they end up "eating" the middle class by
  buying up existing assets and driving up prices.

* **Asset Price Inflation is a Warning Sign:** The speaker warns that rising
  asset prices (like housing) are not a sign of individual wealth but rather a
  symptom of the middle class being dispossessed.

* **The Middle Class is Disappearing:** The speaker predicts that if current
  trends continue, there will be no middle class in the future.

* **Taxation is Key:** The speaker proposes that changing the tax system is
  crucial to address this issue.

**Overall, the video presents a pessimistic view of the future under current
economic systems, arguing that compound interest perpetuates wealth inequality
and threatens the middle class.**

## Transcript

**[00:00:00] SPEAKER:** okay welcome back to Gary's economics today we are
going to explain compound interest okay so compound interest is a very popular
topic on social media especially the financial influencer space famously
Albert Einstein is said to have called compound interest the most powerful
force in the universe he probably didn't really say that what else are we
misunderstanding about compound interest

**[00:00:24]** OK, we should probably start with a very clear explanation of
what compound interest is, why it's so popular and why it's a big deal, why
Albert Einstein allegedly thought it was so important. So the big thing about
compound interest is that it grows in what mathematicians would call an
exponential fashion.

**[00:00:46]** So let's assume you have a really good investment and you're
able to make 10% a year, which is a very high rate of return. Let's say you
start with £1,000. In your first year, you will get given £100 in interest,
let's say. Puts you to £1,100. But in the next year, since you not only get
10% on your initial £1,000, but also this £100, you end up seeing your 1100
go up by not just 100 but 110 pounds which is of course more which means you
end up with 1210 and this process increases each time...

[...truncated for sample; the real file in output/ contains the full transcript]
```

---

## What to notice

- **YAML frontmatter** records all metadata needed for traceability: source file, duration, detected language, processing timestamp, and the exact model versions used.
- **Summary section** uses the language detected from the audio (`## Summary` for English, `## Resumé` for Danish) and gives a structured 5-8 bullet overview with a closing synthesis.
- **Transcript section** uses bold speaker+timestamp prefixes on speaker changes and lone timestamps for continuations from the same speaker — keeps the file scannable but uncluttered.
- **No re-flowing**: each segment from WhisperX becomes one paragraph. This makes diff-friendly editing easy if you later correct or redact specific moments.

## With diarization enabled

A two-person interview would look like:

```markdown
**[00:00:00] SPEAKER_00:** Tak fordi du tog dig tid til at snakke med mig.
**[00:00:04] SPEAKER_01:** Det var så lidt — fortæl mig hvad du vil vide.
**[00:00:07] SPEAKER_00:** Lad os starte fra begyndelsen. Hvornår startede du her?
**[00:00:12] SPEAKER_01:** Det må have været omkring 2018. Måske 2017, jeg er ikke helt sikker.
**[00:00:18]** Men jeg husker tydeligt at det var lige efter omstruktureringen.
```

Note how `SPEAKER_01`'s second utterance (continuing the same speaker) drops the speaker label and shows only the timestamp.
