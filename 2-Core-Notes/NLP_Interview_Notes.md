# 🧠 NLP Interview Notes — From Bag of Words to LLMs

> A single-file, interview-ready reference. Every section follows the same rhythm:
> **Intuition → Math (only what's asked) → Tiny worked example → Code → Interview trap.**
>
> Last updated: August 2026 · Format: GitHub-flavoured Markdown (math renders on GitHub)

---

## 📑 Table of Contents

| # | Section | Why interviewers ask it |
|---|---------|------------------------|
| 0 | [The Big Picture](#0-the-big-picture) | "Walk me through the evolution of NLP" |
| 1 | [Text Preprocessing](#1-text-preprocessing--the-unglamorous-90) | Warm-up + tests real-world experience |
| 2 | [Sparse Representations](#2-sparse-representations-one-hot--bow--tf-idf) | BoW/TF-IDF are *guaranteed* to come up |
| 3 | [Count-Based Semantics](#3-count-based-semantics-co-occurrence-ppmi-lsa) | Bridges "old" and "new" |
| 4 | [Word2Vec](#4-word2vec--prediction-based-embeddings) | The #1 asked embedding question |
| 5 | [GloVe](#5-glove--global-vectors) | Always asked as "how is it different from W2V?" |
| 6 | [FastText](#6-fasttext--subword-embeddings) | The OOV answer |
| 7 | [Comparison & Shared Limitations](#7-comparison--shared-limitations) | Sets up the Transformer pivot |
| 8 | [Python: Build Them All](#8-python-build-them-all) | Coding round |
| 9 | [Sentence & Document Embeddings](#9-sentence--document-embeddings) | Semantic search / RAG roles |
| 10 | [RNN → LSTM → Seq2Seq → Attention](#10-rnn--lstm--seq2seq--attention) | "Why did we need Transformers?" |
| 11 | [The Transformer, In Depth](#11-the-transformer-in-depth) | Core of every modern interview |
| 12 | [Subword Tokenization](#12-subword-tokenization-bpe-wordpiece-unigram) | Very common follow-up |
| 13 | [Pretrained Model Zoo](#13-pretrained-model-zoo-bert-gpt-t5--friends) | "BERT vs GPT?" |
| 14 | [Fine-Tuning & PEFT](#14-fine-tuning--peft-lora-qlora) | Applied ML roles |
| 15 | [The LLM Era](#15-the-llm-era-decoding-prompting-rag-agents-alignment) | 2026 must-know |
| 16 | [Evaluation Metrics](#16-evaluation-metrics) | Where candidates get sloppy |
| 17 | [Core NLP Tasks](#17-core-nlp-tasks) | Applied/system-design |
| 18 | [Production & System Design](#18-production--system-design) | Senior roles |
| 19 | [2025–2026 Trends](#19-20252026-trends) | "What have you read recently?" |
| 20 | [❓ Interview FAQ (60+ Q&A)](#20--interview-faq) | Rapid-fire round |
| 21 | [🎯 Last-Minute Cheat Sheet](#21--last-minute-cheat-sheet) | Night before |

---

## 0. The Big Picture

Before any detail, be able to draw **this** on a whiteboard. Interviewers love a candidate with a map.

```text
                    THE CENTRAL PROBLEM OF NLP
        "Computers do linear algebra. Language is discrete symbols."
                 → How do we turn text into numbers?

  ┌────────────────────────────────────────────────────────────────────┐
  │  ERA 1: SYMBOLIC / SPARSE          (1950s–2000s)                   │
  │  One-Hot → Bag of Words → n-grams → TF-IDF → BM25                  │
  │  ✅ interpretable, fast, no training   ❌ no meaning, huge & sparse │
  └────────────────────────────────────────────────────────────────────┘
                                  ↓  "words in similar contexts mean similar things"
  ┌────────────────────────────────────────────────────────────────────┐
  │  ERA 2: STATIC DENSE EMBEDDINGS    (2003–2016)                     │
  │  LSA/SVD → Word2Vec → GloVe → FastText                             │
  │  ✅ semantics, small dims           ❌ ONE vector per word (polysemy)│
  └────────────────────────────────────────────────────────────────────┘
                                  ↓  "meaning depends on the sentence"
  ┌────────────────────────────────────────────────────────────────────┐
  │  ERA 3: CONTEXTUAL / SEQUENTIAL    (2014–2018)                     │
  │  RNN → LSTM/GRU → Seq2Seq → Attention → ELMo                       │
  │  ✅ context-aware                   ❌ sequential = slow, long-range loss│
  └────────────────────────────────────────────────────────────────────┘
                                  ↓  "attention is all you need"
  ┌────────────────────────────────────────────────────────────────────┐
  │  ERA 4: TRANSFORMERS & PRETRAINING (2017–2020)                     │
  │  Transformer → BERT / GPT-2 / T5 / BART → fine-tuning              │
  │  ✅ parallel, transfer learning     ❌ O(n²), needs task-specific heads│
  └────────────────────────────────────────────────────────────────────┘
                                  ↓  "just scale it and prompt it"
  ┌────────────────────────────────────────────────────────────────────┐
  │  ERA 5: LLMs & BEYOND              (2020–2026)                     │
  │  GPT-3/4 → instruction tuning → RLHF/DPO → RAG → agents →          │
  │  MoE, reasoning models, hybrid Mamba-attention, multimodal          │
  │  ✅ zero-shot everything            ❌ cost, hallucination, alignment │
  └────────────────────────────────────────────────────────────────────┘
```

**The one-liner to memorise:**
> "NLP's history is one long fight against sparsity, then against static meaning, then against sequential computation, and now against cost and hallucination."

### The three properties every representation is judged on

| Property | Question it answers |
|---|---|
| **Sparsity vs Density** | How many dimensions, how many are non-zero? |
| **Static vs Contextual** | Does `bank` change vector by sentence? |
| **Closed vs Open vocabulary** | What happens to a word never seen in training? |

Put any technique on this 3-axis grid and you can answer 80% of representation questions.

---

## 1. Text Preprocessing — the unglamorous 90%

### 🎯 Intuition
Raw text is noisy. Preprocessing decides **what counts as a "unit of meaning"**. Every choice here silently changes your vocabulary size, your feature matrix, and your accuracy.

### The classic pipeline

```text
Raw text
   ↓ 1. Cleaning        strip HTML, URLs, emojis, control chars
   ↓ 2. Normalisation   lowercase, Unicode NFKC, expand contractions, accents
   ↓ 3. Tokenisation    split into words / subwords
   ↓ 4. Stopword removal  (optional! see trap below)
   ↓ 5. Stemming / Lemmatisation  (optional!)
   ↓ 6. n-gram construction
Feature-ready tokens
```

### Tokenisation flavours

| Level | Example: `"don't run!"` | Pros | Cons |
|---|---|---|---|
| **Whitespace** | `["don't", "run!"]` | trivial | punctuation glued on |
| **Rule/regex (word)** | `["do", "n't", "run", "!"]` | clean, interpretable | language-specific, OOV |
| **Character** | `["d","o","n",...]` | zero OOV | very long sequences, weak semantics |
| **Subword (BPE/WordPiece)** | `["don", "'", "t", "run", "!"]` | balances both — **the modern default** | less human-readable |

> 💡 **Why subword won:** vocab of ~30k–200k covers *any* string, including typos, code, and 100 languages, with no `[UNK]`. See [§12](#12-subword-tokenization-bpe-wordpiece-unigram).

### Stemming vs Lemmatisation

| | **Stemming** | **Lemmatisation** |
|---|---|---|
| Method | crude suffix chopping (Porter, Snowball) | dictionary + POS-aware |
| `studies` → | `studi` | `study` |
| `better` → | `better` | `good` |
| `caring` → | `car` ❌ | `care` ✅ |
| Speed | very fast | slower (needs POS tagging) |
| Output is a real word? | Not necessarily | Yes |

**When to use which:** stemming for large-scale search/IR where recall matters more than readability; lemmatisation for linguistic analysis, chatbots, and anywhere output is shown to humans. **With Transformers, use neither** — subword tokenizers handle morphology and the model learns that `run/running/ran` are related.

### 🪤 Interview traps

- **"Should you always remove stopwords?"** → *No.* They're noise for topic classification, but critical for sentiment (`not good`), NLI, QA, and anything Transformer-based. Removing "not" flips your label.
- **"Should you always lowercase?"** → *No.* `US` vs `us`, `Apple` vs `apple`. NER and cased BERT models depend on case.
- **Fit on train only.** `vectorizer.fit_transform(X_train)` then `vectorizer.transform(X_test)`. Fitting on the full corpus leaks IDF statistics from the test set — a classic red flag in interviews.

```python
import re, unicodedata

def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " <URL> ", text)
    text = re.sub(r"<[^>]+>", " ", text)              # HTML
    text = re.sub(r"[^\w\s'!?.,]", " ", text)         # keep sentiment punctuation
    return re.sub(r"\s+", " ", text).strip().lower()

clean("Check <b>THIS</b> out!! https://x.com  — it's GREAT 😀")
# "check this out!! <url> it's great"
```

---
## 2. Sparse Representations: One-Hot → BoW → TF-IDF

### 2.1 One-Hot Encoding

**Intuition:** give every word in the vocabulary its own switch. Word *i* = a vector of zeros with a single 1 at index *i*.

```text
Vocabulary: [apple, bank, river, loan]   (V = 4)

apple → [1, 0, 0, 0]
bank  → [0, 1, 0, 0]
river → [0, 0, 1, 0]
```

**Fatal flaw — everything is equally unrelated.** Cosine similarity between *any* two distinct one-hot vectors is **exactly 0**:

$$\cos(\text{bank}, \text{loan}) = \frac{0}{1 \cdot 1} = 0 = \cos(\text{bank}, \text{banana})$$

The vectors are **mutually orthogonal**, so the geometry carries no semantics. It's just an index in disguise.

> 🎤 **Say this in the interview:** "One-hot encoding is a lookup table pretending to be a vector space. Its dimensionality equals the vocabulary size, and the distance between every pair of words is identical — so no downstream model can learn that *king* and *queen* are related from the representation alone."

---

### 2.2 Bag of Words (BoW)

**Intuition:** shake a document until word order falls out. Keep only *which* words appeared and *how many times*.

**Variants:**

| Variant | Cell value | Use when |
|---|---|---|
| **Binary** | 1 if word present | short texts, presence matters more than frequency |
| **Count** | raw frequency | Naive Bayes, topic-ish tasks |
| **Frequency (normalised)** | count / doc length | documents differ wildly in length |
| **TF-IDF** | count × rarity | **default for classical IR/classification** |

**Worked example**

```text
D1: "the bank of the river was muddy"
D2: "the bank approved the cash loan"

Stopwords removed: {the, of, was}
Vocabulary (alphabetical): [approved, bank, cash, loan, muddy, river]

           approved  bank  cash  loan  muddy  river
D1 (bin)      0       1     0     0      1      1
D2 (bin)      1       1     1     1      0      0
```

**What BoW destroys — the killer example:**

```text
"The dog bit the man."   →  {bit:1, dog:1, man:1, the:2}
"The man bit the dog."   →  {bit:1, dog:1, man:1, the:2}
                            ↑ IDENTICAL VECTORS, opposite meaning
```

### 2.3 n-grams — patching word order

Treat contiguous windows as single features.

```text
"not good at all"
  unigrams: [not, good, at, all]
  bigrams : [not_good, good_at, at_all]        ← "not_good" saves sentiment!
  trigrams: [not_good_at, good_at_all]
```

**The trade-off (memorise this):** n-grams buy local order at the cost of an **exponential vocabulary explosion** and worse sparsity. Unigram vocab ≈ 50k; bigram vocab can be millions. Standard practice: `ngram_range=(1,2)` plus `min_df` to prune rare grams. Eg. across multiple sentences like I love cats, I love dogs ,I love fish ,cats love fish ..unigrams just stores I love cats dogs fish . But bigrams stores I_love, love_cats,love_dogs,love_fish,cats_love,love_fish etc. it explodes as sentences increase

> 🪤 **Trap:** "How do you capture 'New York' as one concept?" → bigrams, or collocation detection (`gensim.models.Phrases` using PMI thresholds), or just use a subword Transformer.

---

### 2.4 TF-IDF — the workhorse

**Intuition (two questions):**
1. **TF:** *How much does this word matter inside this document?* → often.
2. **IDF:** *How much does this word distinguish this document from all others?* → rarely elsewhere.

A word that is **frequent here but rare globally** is a signal. A word frequent everywhere (`the`) is noise.

$$\text{TF-IDF}(t, d, D) = \text{tf}(t,d) \times \log\frac{N}{\text{df}(t)}$$

**Common TF variants:** raw count, `count/len(d)`, or **log-normalised** $1 + \log(\text{count})$ (dampens the fact that appearing 100× isn't 100× more relevant than 1×).

**scikit-learn's exact formula** (know this — interviewers check):

$$\text{idf}(t) = \ln\!\left(\frac{1+N}{1+\text{df}(t)}\right) + 1 \quad\text{(smooth\_idf=True)}$$

Then each row is **L2-normalised**. The `+1` at the end guarantees terms appearing in every document still get a small non-zero weight instead of being deleted.

**Fully worked example** (N=2, corpus above):

```text
df(bank) = 2  → idf = ln(3/3) + 1 = 1.0000     ← in every doc, downweighted
df(river)= 1  → idf = ln(3/2) + 1 = 1.4055     ← distinctive
df(loan) = 1  → idf = ln(3/2) + 1 = 1.4055

D1 raw:  bank=1.0000, muddy=1.4055, river=1.4055
L2 norm = sqrt(1.0² + 1.4055² + 1.4055²) = 2.2251
D1 final = [0, 0.449, 0, 0, 0.632, 0.632]

D2 raw:  bank=1.0000, approved=cash=loan=1.4055
L2 norm = sqrt(1.0² + 3 × 1.4055²) = 2.6317
D2 final = [0.534, 0.380, 0.534, 0.534, 0, 0]

cosine(D1, D2) = 0.449 × 0.380 = 0.171   ← only "bank" overlaps
```

Notice: **`bank` got the *lowest* weight in both documents** even though it's the topical anchor — because it appears everywhere. That's IDF doing its job, and it's also why TF-IDF can't tell you the two `bank`s mean different things.

### 🪤 TF-IDF interview traps

- **"Why L2-normalise?"** So document length doesn't dominate cosine similarity; long docs otherwise get large-magnitude vectors.
- **"IDF with a single document?"** Undefined/useless — IDF needs a corpus. With `smooth_idf`, everything collapses to 1.0.
- **"TF-IDF vs BM25?"** BM25 (§2.6) adds **term-frequency saturation** and **document-length normalisation** and is the actual production IR baseline. If you say "TF-IDF is the search baseline," a good interviewer will correct you to BM25.
- **"Is TF-IDF a machine learning model?"** No — it's a deterministic statistical weighting. Nothing is learned by gradient descent.

---

### 2.5 Hashing Vectorizer (the streaming trick)

Instead of storing a vocabulary dict, hash each token into one of `2^20` buckets.

```python
from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(n_features=2**18, alternate_sign=False)
X = hv.transform(["the bank approved the loan"])   # no .fit() needed!
```

| ✅ Pros | ❌ Cons |
|---|---|
| Constant memory, no vocab stored | **Collisions** (two words → same bucket) |
| Works on streaming / online learning | **Not invertible** — can't recover feature names |
| No fit step, trivially parallel | No IDF weighting (add `TfidfTransformer`) |

Use it when vocabulary is unbounded (logs, tweets firehose) or memory is capped.

---

### 2.6 BM25 — what search engines actually use

$$\text{BM25}(q,d)=\sum_{t \in q} \text{IDF}(t)\cdot\frac{f(t,d)\cdot(k_1+1)}{f(t,d)+k_1\left(1-b+b\cdot\frac{|d|}{\text{avgdl}}\right)}$$

**Two intuitions that are the whole answer:**
1. **Saturation ($k_1$, ~1.2–2.0):** the 10th occurrence of "bank" adds far less than the 2nd. TF-IDF grows linearly; BM25 flattens out.
2. **Length normalisation ($b$, ~0.75):** a long document naturally contains more of everything — penalise it proportionally.

BM25 is still the **hybrid-search partner of dense embeddings in 2026 RAG stacks** — it nails exact keywords, error codes, and rare proper nouns that embeddings blur. See [§15.4](#154-rag-retrieval-augmented-generation).

```python
# rank_bm25 — the 5-line production baseline everyone should know
from rank_bm25 import BM25Okapi
corpus = [d.lower().split() for d in ["the bank of the river was muddy",
                                      "the bank approved the cash loan"]]
bm25 = BM25Okapi(corpus)
print(bm25.get_scores("cash loan".split()))   # → [0.0, 1.06...]
```

---

### 2.7 Sparse family — decision table

| Method | Order? | Semantics? | Dim | Trained? | Killer use case |
|---|---|---|---|---|---|
| One-hot | ❌ | ❌ | V | ❌ | categorical input to a NN |
| Binary BoW | ❌ | ❌ | V | ❌ | short-text presence |
| Count BoW | ❌ | ❌ | V | ❌ | Naive Bayes |
| n-gram BoW | 🟡 local | ❌ | V^n | ❌ | sentiment with negation |
| TF-IDF | ❌ | ❌ | V | ❌ | classic classification, quick baseline |
| Hashing | ❌ | ❌ | fixed | ❌ | streaming / memory-capped |
| BM25 | ❌ | ❌ | V | ❌ | **lexical search, hybrid RAG** |

> 🎤 **Don't dismiss sparse methods.** A TF-IDF + LinearSVC baseline trains in seconds, is fully interpretable, needs no GPU, and on many enterprise classification tasks lands within a few points of a fine-tuned BERT. *Always* mention it as your baseline — interviewers score that as engineering maturity.

---
## 3. Count-Based Semantics: Co-occurrence, PPMI, LSA

### 3.1 The Distributional Hypothesis — the idea behind *all* embeddings

> **"You shall know a word by the company it keeps."** — J.R. Firth, 1957

If `dog` and `cat` both frequently appear near `pet`, `vet`, `feed`, `fur`, then whatever `dog` means must be close to whatever `cat` means. **Meaning ≈ distribution over contexts.** Word2Vec, GloVe, and even BERT are all engineering answers to this single sentence.

### 3.2 Co-occurrence matrix

Build a `V × V` matrix where cell `(i,j)` = how often word *j* appeared within a ±k window of word *i*.

```text
Corpus: "I like deep learning. I like NLP. I enjoy flying."
Window = 1

        I  like  enjoy  deep  learning  NLP  flying
I       0    2      1     0      0       0     0
like    2    0      0     1      0       1     0
enjoy   1    0      0     0      0       0     1
deep    0    1      0     0      1       0     0
```

Each **row is already a word vector** — and unlike one-hot, `like` and `enjoy` now share the `I` dimension, so cosine similarity > 0. Real semantics, from pure counting.

**Problems:** V × V is enormous, still sparse, and raw counts are dominated by frequent words.

### 3.3 PMI and PPMI — fixing the frequency bias

Raw counts say `the` co-occurs with everything. **Pointwise Mutual Information** asks: do these two words co-occur *more than chance*?

$$\text{PMI}(w,c)=\log\frac{P(w,c)}{P(w)P(c)} \qquad \text{PPMI}(w,c)=\max(\text{PMI}(w,c),\,0)$$

- PMI > 0 → associated more than random (`ice`–`cold`)
- PMI = 0 → independent
- PMI < 0 → unreliable with sparse data, so we **clip to 0** → PPMI

> 🔑 **The connection interviewers love:** Levy & Goldberg (2014) proved that **Skip-Gram with Negative Sampling is implicitly factorising a shifted PPMI matrix.** So "predictive" Word2Vec and "count-based" methods are two roads to the same place. Dropping this fact reliably impresses.

### 3.4 LSA / LSI — SVD on the term-document matrix

**Intuition:** take the giant sparse TF-IDF matrix and compress it to `k` latent "topic" dimensions using truncated SVD. Words that co-occur get merged into shared latent factors, which handles synonymy (`car` ≈ `automobile`).

$$X_{\;m\times n} \approx U_k \Sigma_k V_k^\top$$

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

lsa = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=100))
doc_vectors = lsa.fit_transform(corpus)   # dense 100-d document vectors
```

| ✅ | ❌ |
|---|---|
| Dense, handles synonymy | Latent dimensions aren't interpretable |
| Deterministic, no hyperparameter tuning hell | Full SVD is O(min(m,n)²·max(m,n)) — poor scaling |
| Strong classical topic/IR baseline | Still **one vector per word**, no polysemy |

> **LSA vs LDA (a favourite confusion):** LSA is *linear algebra* (SVD, latent dims can be negative, no probabilistic story). **LDA** is a *generative Bayesian model* — documents are mixtures of topics, topics are distributions over words; outputs are non-negative and interpretable as probabilities.

---

## 4. Word2Vec — Prediction-Based Embeddings

*(Mikolov et al., Google, 2013)*

### 🎯 Intuition
Stop counting; **learn**. Train a tiny neural net on a fake task — "predict a word's neighbours" — then **throw away the output layer and keep the hidden weights**. Those weights *are* the embeddings. The fake task is scaffolding; the by-product is the prize. This is **self-supervised learning** before the term was popular.

### 4.1 The two architectures

```text
CBOW — Continuous Bag of Words          SKIP-GRAM
"predict the word from its context"     "predict the context from the word"

 [the] [___] [approved] [loan]           [___] [bank] [___]
   ↓     ↑        ↓        ↓                ↑     ↓     ↑
   └─────┴────────┴────────┘                └─────┴─────┘
      average → predict "bank"            "bank" → predict each neighbour
```

| | **CBOW** | **Skip-Gram** |
|---|---|---|
| Input → Output | context → centre word | centre word → context |
| Training speed | **Faster** (one prediction per window) | Slower (2×window predictions) |
| Rare words | Smoothed away by averaging → weaker | **Much better** (each rare word gets its own gradient updates) |
| Small corpora | Needs more data | **Better** |
| Default choice | large corpus, need speed | **generally preferred (`sg=1`)** |

**Memory hook:** *CBOW = Context ➜ One word. Skip-Gram = Single word ➜ Group.*

### 4.2 The objective (Skip-Gram)

Maximise the log-probability of context words given the centre word:

$$\frac{1}{T}\sum_{t=1}^{T}\sum_{-c \le j \le c,\, j \ne 0} \log p(w_{t+j} \mid w_t)$$

with the softmax

$$p(o \mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w=1}^{V}\exp(u_w^\top v_c)}$$

where $v$ = input ("centre") vectors, $u$ = output ("context") vectors. **Every word gets two vectors during training**; the input matrix is usually what you keep (some implementations average both).

🚨 **The problem:** that denominator sums over the entire vocabulary (millions of words) **for every training pair**. Computationally hopeless. Hence the two famous tricks:

### 4.3 Negative Sampling (the trick that made Word2Vec work)

**Intuition:** don't normalise over the whole vocabulary. Reframe as **binary classification**: "is this (centre, context) pair real, or did I make it up?" Push real pairs together, push `k` random fake pairs apart.

$$\log \sigma(u_o^\top v_c) \;+\; \sum_{i=1}^{k}\mathbb{E}_{w_i \sim P_n(w)}\big[\log \sigma(-u_{w_i}^\top v_c)\big]$$

- `k` = 5–20 for small datasets, 2–5 for large ones.
- Negatives are drawn from the **unigram distribution raised to the 3/4 power**:

$$P_n(w) = \frac{U(w)^{3/4}}{Z}$$

> 🪤 **"Why the 3/4 exponent?"** It's a smoothing knob. Raising to 3/4 *shrinks* the sampling probability of ultra-frequent words (`the`) and *boosts* rare ones relative to plain unigram sampling — so rare words actually appear as negatives often enough to get useful gradient signal. It was found empirically, not derived.

### 4.4 Hierarchical Softmax (the alternative)

Arrange the vocabulary as a **Huffman binary tree**; predicting a word = a sequence of binary decisions down the tree. Cost drops from **O(V) → O(log V)**.

| | Negative Sampling | Hierarchical Softmax |
|---|---|---|
| Best for | frequent words, low dims, **most common choice** | **rare words**, large vocabularies |
| Complexity | O(k) | O(log V) |
| gensim flag | `negative=5, hs=0` | `hs=1, negative=0` |

### 4.5 Subsampling of frequent words

Randomly discard frequent tokens during training with probability

$$P(\text{discard } w_i)=1-\sqrt{\frac{t}{f(w_i)}}, \qquad t \approx 10^{-5}$$

Two benefits: massive speedup, **and** it effectively *widens* the context window around rare words (because the `the`s between them get deleted), improving rare-word quality.

### 4.6 Key hyperparameters (be ready to justify each)

| Param | Typical | Effect |
|---|---|---|
| `vector_size` | 100–300 | more dims = more capacity, more data needed, slower |
| `window` | 2–5 → **syntactic**; 5–10 → **semantic/topical** | ⭐ great answer: small window learns *substitutability* (`good`↔`bad` are both adjectives), big window learns *relatedness* (`good`↔`excellent`, `doctor`↔`hospital`) |
| `min_count` | 5 | drop noisy rare words |
| `negative` | 5–20 | number of noise samples |
| `sg` | 1 = skip-gram, 0 = CBOW | |
| `epochs` | 5–20 | |

### 4.7 The famous vector arithmetic

$$\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$$
$$\vec{Paris} - \vec{France} + \vec{Italy} \approx \vec{Rome}$$

**Why it works:** the training objective encodes relationships as roughly **consistent translation vectors** in the space. The `gender` offset is approximately the same displacement between `king/queen`, `man/woman`, `actor/actress`.

> 🪤 **The honest caveat (say it!):** standard evaluation code **excludes the three input words** from the nearest-neighbour search. Without that exclusion the top hit for `king − man + woman` is often just… `king`. The analogy result is real but weaker than the marketing suggests.

### 4.8 Word2Vec in practice

```python
from gensim.models import Word2Vec

model = Word2Vec(
    sentences=tokenized_corpus,   # list[list[str]]
    vector_size=100, window=5, min_count=5,
    sg=1, negative=10, sample=1e-5,
    epochs=10, workers=8, seed=42,
)
model.wv.most_similar("bank", topn=5)
model.wv.similarity("river", "stream")
model.wv.most_similar(positive=["king", "woman"], negative=["man"])
model.wv.doesnt_match(["breakfast", "lunch", "dinner", "laptop"])   # → 'laptop'

# Continue training on new data (Word2Vec CAN be updated incrementally)
model.build_vocab(new_sentences, update=True)
model.train(new_sentences, total_examples=len(new_sentences), epochs=5)
```

---

## 5. GloVe — Global Vectors

*(Pennington, Socher, Manning; Stanford, 2014)*

### 🎯 Intuition
Word2Vec slides a local window and never sees the corpus as a whole. LSA sees global counts but ignores local structure. **GloVe fuses both**: build the global co-occurrence matrix once, then learn vectors whose **dot product approximates the log of co-occurrence counts**.

### The key insight — *ratios*, not raw probabilities

The meaning of a word is best revealed by **ratios of co-occurrence probabilities** with probe words:

| $P(k \mid \cdot)$ | k = `solid` | k = `gas` | k = `water` | k = `fashion` |
|---|---|---|---|---|
| $P(k \mid ice)$ | large | small | large | small |
| $P(k \mid steam)$ | small | large | large | small |
| **Ratio** | **≫ 1** | **≪ 1** | **≈ 1** | **≈ 1** |

The ratio cleanly separates the discriminative probe (`solid`, `gas`) from the irrelevant/shared ones (`water`, `fashion`). GloVe's whole derivation is: *design a loss so that vector differences encode these ratios.*

### The objective

$$J=\sum_{i,j=1}^{V} f(X_{ij})\left(w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

with the weighting function

$$f(x)=\begin{cases}(x/x_{\max})^{\alpha} & x < x_{\max}\\ 1 & \text{otherwise}\end{cases},\quad \alpha=0.75,\ x_{\max}=100$$

**Why $f(x)$ exists — three reasons, all worth stating:**
1. $f(0)=0$, so zero co-occurrences are skipped (avoids $\log 0$ and the matrix's ~99% zeros).
2. Rare co-occurrences are noisy → downweighted.
3. Ultra-frequent pairs are **capped** so `the–of` doesn't dominate the loss.

### Word2Vec vs GloVe — the comparison question

| | **Word2Vec** | **GloVe** |
|---|---|---|
| Paradigm | Predictive (local windows) | Count-based **matrix factorisation** on global stats |
| Loss | cross-entropy w/ neg. sampling | **weighted least squares** on log-counts |
| Sees corpus | streaming, window by window | one global co-occurrence matrix |
| Training | SGD over pairs, online | AdaGrad over non-zero cells |
| Incremental update | ✅ possible | ❌ must rebuild matrix |
| Memory | low | high (co-occurrence matrix) |
| Empirically | very close; task-dependent | slightly better on analogy benchmarks in original paper |

> 🎤 **The 15-second answer:** "Word2Vec learns from local context windows via prediction; GloVe factorises a global co-occurrence matrix via regression on log counts. Word2Vec is online and memory-light; GloVe uses global statistics and trains faster on a fixed corpus. Both produce **static** vectors, so both share the polysemy and OOV problems."

```python
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-100")   # pretrained, 400k words, 100-d
glove.most_similar("bank", topn=5)
```

---

## 6. FastText — Subword Embeddings

*(Bojanowski et al., Facebook AI, 2016)*

### 🎯 Intuition
Word2Vec treats `run`, `running`, `runner` as three unrelated atoms and dies on unseen words. **FastText represents a word as the sum of its character n-grams**, so morphology is shared and *any* string can be embedded.

```text
"where" with n = 3..6, boundary markers < >:
   <wh, whe, her, ere, re>          (3-grams)
   <whe, wher, here, ere>           (4-grams)
   ... plus the whole-word token <where>

vec("where") = Σ vec(each n-gram)
```

**Why this is the OOV answer:**
```text
"kubernetes" never seen in training?
  → still has n-grams <ku, kub, ube, ber, ern, rne, net, ete, tes, es>
  → many shared with "kubectl", "internet", "network"
  → a meaningful vector instead of [UNK]
```

| ✅ | ❌ |
|---|---|
| **No OOV ever** | Larger model file (n-gram hash table) |
| Excellent for morphologically rich languages (Turkish, Finnish, Hindi, German) | Slower training than Word2Vec |
| Handles typos & domain jargon | Can over-relate words that merely *look* alike (`bank` vs `banking` vs `bankrupt`) |
| Strong, tiny, CPU-only classifier (`fastText supervised`) | Still **static** — polysemy unsolved |

```python
from gensim.models import FastText
ft = FastText(sentences=tokenized_corpus, vector_size=100,
              window=5, min_count=1, sg=1, min_n=3, max_n=6)
ft.wv["kubernetes"]              # ✅ works even if never in training data
"kubernetes" in ft.wv.key_to_index   # may be False — but the vector still exists
```

> 🪤 **Trap:** "FastText solves OOV, so why do we need subword *tokenizers* in Transformers?" → FastText builds a *word* vector from n-grams (bag-of-n-grams, order-free). Transformer tokenizers split the sequence into ordered subword *tokens* that each get contextualised. Different mechanisms, similar motivation.

---

## 7. Comparison & Shared Limitations

### 7.1 Full representation comparison

| Feature | One-Hot | BoW | TF-IDF | LSA | Word2Vec | GloVe | FastText | **BERT** |
|---|---|---|---|---|---|---|---|---|
| Dimensionality | V | V | V | ~100–500 | 100–300 | 100–300 | 100–300 | 768–1024 |
| Dense? | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Semantics? | ❌ | ❌ | ❌ | 🟡 | ✅ | ✅ | ✅ | ✅✅ |
| Word order? | ❌ | ❌ | ❌ | ❌ | 🟡 window | 🟡 window | 🟡 window | ✅ full |
| Handles OOV? | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ subword |
| **Context-dependent?** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | **✅** |
| Training cost | none | none | none | low | medium | medium | medium | very high |
| Interpretable | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

### 7.2 The two bottlenecks that created the Transformer

Word2Vec and GloVe are massive semantic upgrades over BoW and TF-IDF, but they share **two foundational bottlenecks** that paved the way for modern architectures:

**① The Polysemy Bottleneck (static nature)**
Both allocate **exactly one fixed vector per unique string** in the vocabulary lookup table. They cannot represent multi-meaning context natively. `Apple` gets the *identical* vector next to `orchard` and next to `iPhone` — the model is forced to store a blurry average of every sense.

```text
"The bank of the river was muddy."   ─┐
                                      ├─► vec("bank") = [0.21, -0.44, 0.87, ...]  (SAME)
"The bank approved the cash loan."   ─┘
```

**② The Out-of-Vocabulary (OOV) Deficit**
Both are completely blind to tokens unseen at training time. A typo, a novel compound, or a new product name causes the lookup to fail or emit an uninformative `[UNK]` — and adapting requires a **full retraining cycle**. (FastText patches this with n-grams; subword tokenizers solve it properly.)

**Bonus bottleneck worth mentioning:** static embeddings encode and amplify **corpus bias** (`man:computer_programmer :: woman:homemaker`, Bolukbasi et al. 2016). Debiasing is a live research area and a great "responsible AI" talking point.

---
## 8. Python: Build Them All

> This is the cleaned-up, runnable version of the classic "bank homonym" pipeline — the single best 40-line demo to have memorised, because it *proves* the polysemy bottleneck numerically.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# ── 1. Corpus: the 'bank' homonym case study ─────────────────────────────
sentences = [
    "The bank of the river was muddy.",
    "The bank approved the cash loan.",
]
stop_words_list = ["the", "of", "was"]

# ── 2. Sparse counting spaces ────────────────────────────────────────────
binary_vectorizer = CountVectorizer(binary=True, stop_words=stop_words_list)
binary_bow_matrix = binary_vectorizer.fit_transform(sentences).toarray()
vocabulary = binary_vectorizer.get_feature_names_out()

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_list)
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences).toarray()

print("Vocabulary Key:", list(vocabulary))
print("Sentence A (Binary BoW):", binary_bow_matrix[0])
print("Sentence B (Binary BoW):", binary_bow_matrix[1])
print("Sentence A (TF-IDF):", np.round(tfidf_matrix[0], 3))
print("Sentence B (TF-IDF):", np.round(tfidf_matrix[1], 3))

# ── 3. Dense continuous space (Word2Vec) ─────────────────────────────────
tokenized_corpus = [
    [w for w in doc.lower().replace(".", "").split() if w not in stop_words_list]
    for doc in sentences
]

w2v_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=3,   # toy 3-D embedding so we can print it
    window=2,        # context window distance
    min_count=1,     # keep rare tokens (toy corpus!)
    sg=1,            # 1 = Skip-Gram, 0 = CBOW
    seed=42,
    workers=1,       # required for reproducibility with a fixed seed
)

v_bank = w2v_model.wv["bank"]
print("\nWord2Vec Static Vector ['bank']:", np.round(v_bank, 4))

# ── 4. Sentence vector via mean pooling ──────────────────────────────────
def compute_sentence_average(token_list, model):
    vectors = [model.wv[w] for w in token_list if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

sent_a = compute_sentence_average(tokenized_corpus[0], w2v_model)
sent_b = compute_sentence_average(tokenized_corpus[1], w2v_model)
print("Sentence A (pooled avg):", np.round(sent_a, 4))
print("Sentence B (pooled avg):", np.round(sent_b, 4))

# 🔑 THE PUNCHLINE
assert np.array_equal(w2v_model.wv["bank"], v_bank)
print("\nIs vec('bank') identical in BOTH sentences?  →  YES. "
      "This is the polysemy bottleneck.")
```

**Verified output of the sparse section** (scikit-learn 1.8):

```text
Vocabulary Key: ['approved', 'bank', 'cash', 'loan', 'muddy', 'river']
Sentence A (Binary BoW): [0 1 0 0 1 1]
Sentence B (Binary BoW): [1 1 1 1 0 0]
Sentence A (TF-IDF): [0.    0.449 0.    0.    0.632 0.632]
Sentence B (TF-IDF): [0.534 0.38  0.534 0.534 0.    0.   ]

idf: bank=1.0000 (in both docs → downweighted)
     all others=1.4055 (distinctive)
cosine(A, B) = 0.171   ← the ONLY shared signal is the ambiguous word "bank"
```

### ⚠️ Four things a sharp interviewer will point out about this snippet

1. **`vector_size=3` on 7 tokens is a toy.** Real embeddings need ≥ millions of tokens; with two sentences the vectors are essentially noise. Say so before they do.
2. **`min_count=1` is only for demos.** In production `min_count=5` removes noisy hapax legomena.
3. **`workers>1` breaks reproducibility** even with `seed=42` — thread scheduling changes update order. Set `workers=1` for deterministic runs.
4. **Mean pooling throws away word order** — `sent_a` and `sent_b` would be identical if the sentences were anagrams of each other. See [§9](#9-sentence--document-embeddings) for better options.

### 🔁 The contextual contrast (run this to close the loop)

```python
import torch
from transformers import AutoTokenizer, AutoModel

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
mdl = AutoModel.from_pretrained("bert-base-uncased")

def bank_vector(sentence):
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc).last_hidden_state[0]
    idx = tok.convert_ids_to_tokens(enc["input_ids"][0]).index("bank")
    return out[idx]

a = bank_vector("The bank of the river was muddy.")
b = bank_vector("The bank approved the cash loan.")

cos = torch.nn.functional.cosine_similarity(a, b, dim=0)
print(f"BERT cosine('bank' river vs 'bank' money) = {cos:.3f}")
# ≈ 0.4–0.6 — clearly DIFFERENT vectors.
# Word2Vec would print exactly 1.000.
```

> 🎤 **This two-script comparison is the single strongest thing you can demo in an NLP interview.** It shows the problem *and* the solution with numbers, not adjectives.

---

## 9. Sentence & Document Embeddings

Word vectors are not sentence vectors. Here's the ladder, worst → best.

| Method | How | Verdict |
|---|---|---|
| **Mean pooling of W2V** | average all word vectors | Cheap baseline. Loses order; frequent words dominate. |
| **TF-IDF weighted mean** | weight each word vector by its IDF | Noticeably better than plain mean. |
| **SIF / uSIF** | weight by $\frac{a}{a+p(w)}$, then **remove the first principal component** | Surprisingly strong, near-free. The PC-removal step strips the "common discourse" direction. |
| **Doc2Vec (PV-DM / PV-DBOW)** | add a trainable paragraph vector to the W2V objective | Historically important; needs inference-time optimisation for new docs; largely superseded. |
| **[CLS] token of vanilla BERT** | take `last_hidden_state[:,0]` | 🚨 **Bad without fine-tuning** — worse than averaged GloVe on STS. |
| **Sentence-BERT (SBERT)** | **siamese/triplet** fine-tuning with cosine or contrastive loss | ✅ **The correct answer for semantic similarity/search.** |
| **Modern embedding models** | contrastive-trained LLM encoders (E5, BGE, GTE, Qwen3-Embedding, Voyage, OpenAI `text-embedding-3`) | ✅ **2026 production default.** |

### Why vanilla BERT `[CLS]` fails at similarity

BERT was pretrained on MLM + NSP. Nothing in that objective asks embeddings to be **cosine-comparable**. The resulting space is **anisotropic** — all vectors crowd into a narrow cone, so *everything* looks similar (cosine ≈ 0.8 for random sentence pairs). SBERT fixes this by explicitly training with a similarity objective.

### Cross-encoder vs Bi-encoder (⭐ high-frequency question)

```text
BI-ENCODER (SBERT)                    CROSS-ENCODER (reranker)
  A → BERT → vec_A ─┐                   [A [SEP] B] → BERT → score
  B → BERT → vec_B ─┴→ cosine
  
  ✅ Encode corpus ONCE, offline        ✅ Sees full token-level interaction
  ✅ Millisecond ANN search             ❌ Must run the model for EVERY pair
  ❌ Less accurate                      ❌ O(N) per query — cannot pre-index
  → USE FOR: retrieval over millions    → USE FOR: reranking the top ~50
```

**Production pattern:** bi-encoder retrieves top-100 → cross-encoder reranks to top-5. This is the standard 2-stage RAG retriever.

```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# Stage 1 — bi-encoder retrieval
bi = SentenceTransformer("all-MiniLM-L6-v2")          # 384-d, fast, great default
corpus_emb = bi.encode(corpus, normalize_embeddings=True, convert_to_tensor=True)
q_emb = bi.encode("how do I reset my password", normalize_embeddings=True,
                  convert_to_tensor=True)
hits = util.semantic_search(q_emb, corpus_emb, top_k=50)[0]

# Stage 2 — cross-encoder rerank
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [("how do I reset my password", corpus[h["corpus_id"]]) for h in hits]
scores = ce.predict(pairs)
```

### Similarity metrics — know when each applies

| Metric | Formula | Note |
|---|---|---|
| **Cosine** | $\frac{a\cdot b}{\|a\|\|b\|}$ | **Default for text.** Ignores magnitude (doc length). |
| **Dot product** | $a \cdot b$ | Equals cosine **iff vectors are L2-normalised**. Faster. |
| **Euclidean (L2)** | $\|a-b\|_2$ | Monotonically equivalent to cosine on normalised vectors. |
| **Jaccard** | $\frac{|A \cap B|}{|A \cup B|}$ | For *sets* (tokens, shingles), not dense vectors. |

> 🪤 **Trap:** "Is cosine similarity the same as dot product?" → Only after L2 normalisation. Many vector DBs default to dot product; if you forget to normalise, long documents win every query.

---
## 10. RNN → LSTM → Seq2Seq → Attention

You must be able to explain **why each one was invented** — that narrative *is* the interview answer.

### 10.1 Vanilla RNN

**Intuition:** process tokens one at a time, carrying a hidden "memory" state forward.

$$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$$

```text
x₁ ──► [RNN] ──► h₁ ──► [RNN] ──► h₂ ──► [RNN] ──► h₃
        ▲                 ▲                 ▲
       x₁                x₂                x₃    (same weights reused)
```

**The fatal flaw — vanishing/exploding gradients.** Backprop through 50 timesteps multiplies the same Jacobian 50 times. If its dominant eigenvalue < 1, the gradient decays to zero (vanishing → can't learn long dependencies); if > 1, it blows up (exploding → NaNs).

- **Exploding** is easy: **gradient clipping**.
- **Vanishing** is hard: needs an architectural fix → **LSTM**.

### 10.2 LSTM — gates and a highway

**Intuition:** add a **cell state** $C_t$ that flows through the sequence with only *additive* interactions — a "conveyor belt" gradients can travel along without repeated multiplication. Three gates decide what to erase, write, and expose.

$$
\begin{aligned}
f_t &= \sigma(W_f[h_{t-1},x_t]) && \text{\textbf{forget} gate: what to erase from memory}\\
i_t &= \sigma(W_i[h_{t-1},x_t]) && \text{\textbf{input} gate: how much new info to admit}\\
\tilde{C}_t &= \tanh(W_C[h_{t-1},x_t]) && \text{candidate memory}\\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t && \text{\textbf{additive} update ← the magic}\\
o_t &= \sigma(W_o[h_{t-1},x_t]) && \text{\textbf{output} gate}\\
h_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

> 🔑 **The one-sentence answer to "how does LSTM fix vanishing gradients?"**
> "The cell-state update is **additive**, not multiplicative, so gradients flow backwards through $C_t$ largely unattenuated — the forget gate can hold $f_t \approx 1$ and keep a memory alive for hundreds of steps."

### 10.3 GRU

Merges forget+input into a single **update gate** $z_t$ and adds a **reset gate** $r_t$. **2 gates, no separate cell state, ~25% fewer parameters.**

| | LSTM | GRU |
|---|---|---|
| Gates | 3 (f, i, o) + cell state | 2 (update, reset) |
| Params | more | fewer → faster, less overfitting on small data |
| Performance | ≈ equal in practice | ≈ equal in practice |
| Rule of thumb | large data / long sequences | small data / limited compute |

### 10.4 Bidirectional RNNs
Run one RNN left→right and another right→left, concatenate hidden states. Essential for **tagging/classification** (you can see the whole sentence). **Impossible for generation** (you can't see the future you haven't generated). This distinction *is* the encoder-vs-decoder distinction later.

### 10.5 Seq2Seq and the bottleneck

```text
        ENCODER                        DECODER
  x₁→x₂→x₃→x₄ ──► [c] ──► y₁→y₂→y₃
                   ▲
      the ENTIRE source sentence squeezed into ONE fixed vector
```

**The information bottleneck:** a 50-word sentence and a 5-word sentence get the same 512 floats. BLEU scores fell off a cliff as source length grew.

### 10.6 Attention (Bahdanau 2014 / Luong 2015) — the fix

**Intuition:** instead of one frozen summary, let the decoder **look back at all encoder states and take a weighted average, re-weighted at every output step.** A soft, differentiable dictionary lookup.

$$e_{ti}=\text{score}(s_{t-1},h_i),\quad \alpha_{ti}=\frac{\exp(e_{ti})}{\sum_j \exp(e_{tj})},\quad c_t=\sum_i \alpha_{ti}h_i$$

| | **Bahdanau (additive)** | **Luong (multiplicative)** |
|---|---|---|
| Score | $v^\top\tanh(W_1 s + W_2 h)$ | $s^\top W h$ or $s^\top h$ |
| Cost | small MLP per pair | one matmul — **faster** |
| Uses state | $s_{t-1}$ (before output) | $s_t$ (after) |

Attention also gave the first **free interpretability**: plot $\alpha$ as a heatmap and you see soft word alignments.

> 🎤 **The pivot sentence to the next section:** "Attention was invented as a *patch* on the RNN bottleneck. The Transformer's insight was that attention was doing all the work — so remove the RNN entirely."

---

## 11. The Transformer, In Depth

*("Attention Is All You Need", Vaswani et al., 2017 — read it; it will be asked about.)*

### 11.1 Why it won: the complexity table

| Layer type | Complexity per layer | Sequential ops | Max path length |
|---|---|---|---|
| Self-Attention | $O(n^2 \cdot d)$ | **$O(1)$** ✅ | **$O(1)$** ✅ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ ❌ | $O(n)$ ❌ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |

**Two wins, both decisive:**
1. **`O(1)` sequential operations → full GPU parallelism** across the sequence during training. RNNs *cannot* be parallelised over time.
2. **`O(1)` maximum path length** → any token can attend directly to any other token in one hop. No signal decay over 500 tokens.

The cost is $O(n^2)$ memory/compute in sequence length — which is exactly what every efficiency paper since 2019 attacks.

### 11.2 Scaled Dot-Product Attention

For each token, produce three linear projections:
- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I advertise about myself?"
- **Value (V)** — "what do I actually contribute if attended to?"

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**The database analogy (use this — it lands every time):** it's a **soft dictionary lookup**. In `dict[query]` you match one key exactly. Here you compute a similarity between the query and *every* key, softmax it into weights, and return a **weighted blend of all the values**.

> 🪤 **"Why divide by $\sqrt{d_k}$?"** — asked constantly.
> If $q, k$ have i.i.d. components with mean 0 and variance 1, then $q\cdot k$ has variance $d_k$. For $d_k = 64$, dot products swing to ±8 or more, pushing softmax into a near-one-hot regime where **gradients vanish**. Dividing by $\sqrt{d_k}$ restores unit variance and keeps the softmax in its high-gradient region.

### 11.3 Multi-Head Attention

Run $h$ attention operations in parallel with $d_k = d_{\text{model}}/h$, concatenate, project:

$$\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O$$

**Why?** One softmax produces one averaged view of the sentence. Multiple heads let the model attend to different **representation subspaces** simultaneously — empirically, some heads track syntactic dependencies, some track coreference, some just attend to the previous token or to `[SEP]` (the "no-op" heads). Note that total compute is roughly unchanged because each head is $d_{\text{model}}/h$ wide.

### 11.4 The three attention types in the original architecture

| Type | Where | Q, K, V come from | Masked? |
|---|---|---|---|
| **Encoder self-attention** | encoder | all from encoder input | ❌ bidirectional |
| **Masked (causal) self-attention** | decoder | all from decoder input | ✅ future positions set to $-\infty$ |
| **Cross-attention** | decoder | **Q** from decoder, **K/V** from encoder | ❌ |

**Causal masking** is the single mechanism that separates BERT-style from GPT-style models. It's implemented by adding a $-\infty$ upper-triangular matrix to the scores *before* softmax, making those weights exactly 0.

### 11.5 The full block

```text
                    ┌─────────────────────────────┐
   Input Embedding  │  + Positional Encoding      │
        + PE        └──────────────┬──────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │  Multi-Head Self-Attention   │
                    └──────────────┬───────────────┘
                                   ▼
                       Add & Norm  (residual + LayerNorm)
                                   ▼
                    ┌──────────────────────────────┐
                    │  Feed-Forward Network        │
                    │  Linear(d→4d) → GELU →       │
                    │  Linear(4d→d)                │
                    └──────────────┬───────────────┘
                                   ▼
                       Add & Norm            × N layers
```

**Component quiz answers:**

- **Residual connections** — gradient highway; without them a 12-layer Transformer won't train.
- **LayerNorm (not BatchNorm!)** — normalises across the *feature* dimension per token, so it's independent of batch size and of variable sequence length. BatchNorm's per-batch statistics are unstable with padded, variable-length text. Modern models use **RMSNorm** (drops the mean-centering, cheaper).
- **Post-LN vs Pre-LN** — the original paper put LayerNorm *after* the residual add (Post-LN), which needs learning-rate warmup to train. **Pre-LN** (`x + Attn(LN(x))`) is far more stable and is what essentially all modern LLMs use.
- **The FFN** — a position-wise 2-layer MLP with a **4× expansion**. It holds roughly **⅔ of the model's parameters** and is widely interpreted as the model's key-value *knowledge store*. It's also exactly what MoE replaces with sparse experts.
- **Activation** — ReLU originally; now **GELU / SwiGLU**.

### 11.6 Positional encodings — because attention is permutation-invariant

Self-attention is a **set** operation: shuffle the tokens and the output is shuffled identically. Word order must be injected explicitly.

$$PE_{(pos,2i)}=\sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad PE_{(pos,2i+1)}=\cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

| Scheme | Used by | Idea | Extrapolates? |
|---|---|---|---|
| **Sinusoidal (absolute)** | original Transformer | fixed sin/cos of different wavelengths; $PE_{pos+k}$ is a linear function of $PE_{pos}$ | 🟡 somewhat |
| **Learned absolute** | BERT, GPT-2 | a trainable embedding per position | ❌ hard cap at max_len |
| **RoPE (rotary)** | **LLaMA, Qwen, Mistral, most 2024–26 LLMs** | *rotate* Q and K by an angle proportional to position → dot product depends on **relative** distance | ✅ + extendable via NTK/YaRN scaling |
| **ALiBi** | BLOOM, MPT | add a linear distance penalty to attention scores | ✅ excellent |
| **NoPE** | some decoder-only research | causal masking alone leaks position info | 🟡 |

> 🎤 **Modern answer:** "RoPE, because it encodes *relative* position directly in the attention dot product and can be extended past the training context length with interpolation tricks like YaRN — which is how models jumped from 4k to 128k–1M context."

### 11.7 Efficiency: how the O(n²) got tamed

| Technique | Idea |
|---|---|
| **FlashAttention (1/2/3)** | Not an approximation — an **IO-aware exact** kernel. Tiles the computation in SRAM so the $n\times n$ matrix is never written to HBM. Memory becomes O(n). |
| **Multi-Query / Grouped-Query Attention (MQA/GQA)** | Share K/V heads across query heads → shrinks the **KV cache** dramatically at inference. GQA is the standard middle ground. |
| **Sliding-window / local attention** | Each token attends to a local window (Mistral, Longformer) — O(n·w). |
| **Sparse / linear attention** | Longformer, BigBird, Performer, Linformer. |
| **KV caching** | At generation time, cache past K/V so each new token is O(n) not O(n²). *The* reason inference is feasible. |
| **PagedAttention (vLLM)** | Virtual-memory-style paging of the KV cache → huge throughput gains. |
| **Hybrid SSM-attention** | Interleave Mamba-2 / Gated DeltaNet layers with attention (Nemotron-3, Qwen3.6) — see [§19](#19-20252026-trends). |

### 11.8 The Transformer resolution: solving polysemy (concretely)

```text
Static Lookups:      Token "bank" ──────────────────► one fixed vector, forever

Transformer Stack:   context tokens ─► Self-Attention ─► dynamically weighted vector
```

**Sentence A** — parsing `bank`, its **query** vector matches highly against the **keys** for `river` and `muddy`. The attention distribution puts most mass on geographic **values**.
**Sentence B** — the same query now matches `approved` and `loan`; the weighted sum shifts into financial feature space.

**Result:** the output vector for `bank` is computed **at runtime** and shares no fixed coordinates between the two sentences. Polysemy is resolved not by having many vectors per word, but by having **no stored vector at all** — only a function of context.

**And OOV is resolved by subword tokenization:**

$$\text{"riverbank"}\rightarrow[\text{"river"},\text{"\#\#bank"}]\qquad \text{"loanshark"}\rightarrow[\text{"loan"},\text{"\#\#shark"}]$$

Because `river`, `bank`, `loan` all exist in the vocabulary, the model preserves core semantics for a string it has never seen, instead of emitting `[UNK]`.

---
## 12. Subword Tokenization: BPE, WordPiece, Unigram

### 🎯 Intuition
Word-level vocab → OOV disaster. Character-level → sequences 5× longer and weak semantics. **Subword is the compromise:** keep frequent words whole, split rare words into meaningful pieces.

```text
"unhappiness"      → ["un", "happi", "ness"]      (morphology preserved)
"tokenization"     → ["token", "ization"]
"Kubernetes"       → ["Kub", "ernet", "es"]        (never seen? still fine)
```

### 12.1 Byte-Pair Encoding (BPE) — GPT, LLaMA, RoBERTa

**Algorithm (be ready to state it in 4 lines):**
1. Start with a vocabulary of all individual characters (or **bytes** — see below).
2. Count all adjacent symbol pairs in the corpus.
3. **Merge the most frequent pair** into a new symbol; record the merge rule.
4. Repeat until the vocabulary reaches the target size (e.g. 50k).

**Tiny trace:**
```text
Corpus counts:  low:5  lower:2  newest:6  widest:3

Start: l o w </w> | l o w e r </w> | n e w e s t </w> | w i d e s t </w>

Pair frequencies →  "e s" = 9   ← most frequent
Merge 1: e s → es          n e w es t | w i d es t
Merge 2: es t → est        n e w est  | w i d est
Merge 3: est </w> → est</w>
Merge 4: l o → lo
Merge 5: lo w → low        low | low e r
...
```
Encoding a new word applies the learned merges **in the same order**.

**Byte-level BPE (GPT-2 onwards):** operate on the 256 raw **bytes** instead of Unicode characters. Guarantees **zero OOV for any possible input on Earth** — emoji, Chinese, binary garbage, code — with a base vocab of only 256.

### 12.2 WordPiece — BERT, DistilBERT, ELECTRA

Same greedy merge loop, but the merge criterion is **likelihood**, not raw frequency:

$$\text{score}(A,B)=\frac{\text{freq}(AB)}{\text{freq}(A)\times\text{freq}(B)}$$

> 🪤 **"BPE vs WordPiece — what's the actual difference?"** BPE merges the **most frequent** pair; WordPiece merges the pair that **most increases the likelihood of the training data**, which is equivalent to normalising by how frequent the individual pieces already are. Consequence: WordPiece avoids merging a rare token with a very common one just because the common one is everywhere. Notation differs too — WordPiece marks continuations with `##` (`##bank`), byte-BPE marks word *starts* with `Ġ`.

### 12.3 Unigram LM & SentencePiece — T5, ALBERT, XLNet, LLaMA

**Unigram works backwards:** start with a *large* candidate vocabulary, then **iteratively prune** the tokens whose removal least hurts corpus likelihood (EM algorithm). Because it keeps a probability per token, it can produce *multiple* valid segmentations and sample among them — enabling **subword regularisation** (a data-augmentation trick).

**SentencePiece** is the *library/wrapper*, not an algorithm (it can run BPE or Unigram). Its key contribution: treat input as a **raw stream including whitespace**, encoding spaces as `▁`. This makes it **language-agnostic** (works for Chinese/Japanese/Thai with no spaces) and **losslessly reversible** — `detokenize(tokenize(x)) == x` exactly.

### 12.4 Summary table

| Algorithm | Direction | Criterion | Used by |
|---|---|---|---|
| **BPE** | bottom-up merge | max pair frequency | GPT-2/3/4, RoBERTa, LLaMA |
| **WordPiece** | bottom-up merge | max likelihood gain | BERT, DistilBERT, ELECTRA |
| **Unigram** | top-down prune | min likelihood loss | T5, ALBERT, XLNet |
| **SentencePiece** | *wrapper* for BPE/Unigram | — | T5, LLaMA, mBART |

### 12.5 Practical consequences interviewers probe

- **Token ≠ word.** English ≈ **0.75 words per token** (~4 chars/token). This is your cost and context-length math.
- **Non-English costs more.** The same sentence in Hindi/Thai/Burmese can consume 2–5× more tokens on an English-centric tokenizer → higher API cost, less usable context. This is a real fairness issue.
- **Numbers tokenize badly.** `1234` may split as `12|34`, which is a known root cause of arithmetic errors. Modern models often force **digit-level splitting**.
- **Never mix tokenizer and model.** Always `AutoTokenizer.from_pretrained(same_checkpoint)`.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tok.tokenize("Tokenization of riverbank and loanshark"))
# ['token', '##ization', 'of', 'river', '##bank', 'and', 'loan', '##shar', '##k']
```

---

## 13. Pretrained Model Zoo: BERT, GPT, T5 & Friends

### 13.1 The three families — the mental model

```text
┌──────────────────┬──────────────────────┬──────────────────────┐
│ ENCODER-ONLY     │ DECODER-ONLY         │ ENCODER-DECODER      │
│ (autoencoding)   │ (autoregressive)     │ (seq2seq)            │
├──────────────────┼──────────────────────┼──────────────────────┤
│ BERT, RoBERTa,   │ GPT-2/3/4, LLaMA,    │ T5, BART, mT5,       │
│ DeBERTa, ELECTRA │ Mistral, Qwen, Gemma │ Flan-T5, mBART       │
├──────────────────┼──────────────────────┼──────────────────────┤
│ Bidirectional    │ Causal (left→right)  │ Bidir. enc + causal  │
│ attention        │ masked attention     │ dec + cross-attn     │
├──────────────────┼──────────────────────┼──────────────────────┤
│ 🎯 UNDERSTANDING │ 🎯 GENERATION        │ 🎯 TRANSDUCTION      │
│ classification,  │ chat, completion,    │ translation,         │
│ NER, extractive  │ code, reasoning,     │ summarization,       │
│ QA, embeddings   │ everything zero-shot │ structured rewriting │
└──────────────────┴──────────────────────┴──────────────────────┘
```

> 🎤 **"BERT vs GPT" in one breath:** "BERT is an encoder trained with masked language modelling, so every token sees both left and right context — great for understanding, useless for generation. GPT is a decoder trained with causal language modelling, so each token only sees the past — that constraint is what makes it able to generate."

### 13.2 BERT — the details you'll be quizzed on

**Pretraining objective 1 — Masked Language Modelling (MLM):** mask **15%** of tokens, predict them. The 15% is split:
- **80%** → replaced with `[MASK]`
- **10%** → replaced with a **random** token
- **10%** → **left unchanged**

> 🪤 **"Why the 80/10/10 split?"** Because `[MASK]` never appears at fine-tuning/inference time — a **pretrain–finetune mismatch**. Occasionally showing real or random tokens forces the model to build a good contextual representation of *every* token, not just of `[MASK]` positions.

**Pretraining objective 2 — Next Sentence Prediction (NSP):** binary classify whether sentence B follows sentence A. **RoBERTa showed NSP is basically useless** and removing it improved results.

**Special tokens:** `[CLS]` (pooled sequence representation), `[SEP]` (separator), `[PAD]`, `[UNK]`, `[MASK]`.
**Input embedding = token + segment + position embeddings**, summed.
**Sizes:** BERT-base = 12 layers, 768 hidden, 12 heads, **110M** params, 512-token max. BERT-large = 24/1024/16, **340M**.

### 13.3 The BERT successors — one line each

| Model | The one change that matters |
|---|---|
| **RoBERTa** | Drop NSP, **dynamic masking** (new mask each epoch), 10× data, bigger batches, byte-BPE. *"BERT was undertrained."* |
| **ALBERT** | **Cross-layer parameter sharing** + **factorised embedding** ($V\times E$, $E\times H$ with $E \ll H$); NSP → **SOP** (sentence *order* prediction, a harder task). Fewer params, not necessarily faster inference. |
| **DistilBERT** | **Knowledge distillation**: 6 layers, **40% smaller, 60% faster, ~97% of BERT's GLUE score**. The go-to for latency-bound production. |
| **ELECTRA** | Replaces MLM with **replaced-token detection** — a small generator corrupts tokens, the discriminator classifies *every* token as real/fake. Learns from **100% of tokens instead of 15%** → far more sample-efficient. |
| **DeBERTa (v3)** | **Disentangled attention** (separate content and position vectors) + enhanced mask decoder. Long-time SOTA on SuperGLUE; still an excellent encoder in 2026. |
| **Longformer / BigBird** | Sliding-window + global attention → 4k–16k tokens at linear cost. |
| **ModernBERT** | 2024-era encoder refresh: RoPE, GeGLU, 8k context, Flash Attention. **The encoder to reach for now** if you need a small, fast classifier. |
| **SBERT** | Siamese fine-tuning for cosine-comparable **sentence** embeddings. |

### 13.4 T5 and BART

- **T5** — *"Text-to-Text Transfer Transformer."* Every task is cast as text→text with a task prefix: `"translate English to German: ..."`, `"summarize: ..."`. Pretraining is **span corruption** (mask contiguous spans, replace with sentinel tokens `<X>`, `<Y>`; decoder reconstructs them). **Flan-T5** adds massive multi-task instruction tuning — still a superb small workhorse.
- **BART** — a **denoising autoencoder**. Corrupt the input in arbitrary ways (token masking, deletion, **text infilling**, sentence permutation, document rotation), then reconstruct the original. Excellent at summarization.

### 13.5 Decoder-only / LLM lineage (know the arc)

`GPT-1 (pretrain+finetune)` → `GPT-2 (zero-shot emerges)` → `GPT-3 (in-context/few-shot learning, 175B)` → `InstructGPT/ChatGPT (SFT + RLHF)` → `GPT-4 & Claude & Gemini (multimodal, long context)` → `o-series / reasoning models (test-time compute)` → `MoE + hybrid architectures (2025–26)`.

Open-weight lineage: `LLaMA → LLaMA 2/3 → Mistral/Mixtral (MoE) → Qwen 2.5/3 → DeepSeek V3/R1 → Gemma → Nemotron`.

**Architectural deltas from the 2017 paper that essentially every modern LLM shares:**
Pre-LN → **RMSNorm** · learned PE → **RoPE** · ReLU → **SwiGLU** · MHA → **GQA** · dense FFN → **MoE** · no bias terms · **FlashAttention** kernels.

---

## 14. Fine-Tuning & PEFT (LoRA, QLoRA)

### 14.1 The adaptation ladder — pick the cheapest rung that works

| Approach | Params trained | Data needed | When |
|---|---|---|---|
| **Zero-shot prompting** | 0 | 0 | Always try first |
| **Few-shot / in-context** | 0 | 3–30 examples | Quick wins, format control |
| **RAG** | 0 | a document corpus | **Knowledge** problems |
| **Feature extraction** (frozen encoder + logistic regression) | ~1k | 100s | Tiny data, strong baseline |
| **PEFT / LoRA** | 0.1–2% | 500–50k | **Behaviour/format/style/domain** |
| **Full fine-tuning** | 100% | 10k+ | Max quality, big budget |
| **Continued pretraining** | 100% | billions of tokens | New language or deep domain |

> 🎤 **The single most important framing:** **RAG is for *knowledge*; fine-tuning is for *behaviour*.** If the model doesn't *know* your Q3 numbers → RAG. If it knows but won't answer in your required JSON schema/tone/taxonomy → fine-tune. Saying this correctly is often worth more than any architecture detail.

### 14.2 LoRA — Low-Rank Adaptation

**Intuition:** the weight *update* during fine-tuning has low intrinsic rank. So freeze $W$ and learn a cheap low-rank correction.

$$h = W_0x + \Delta Wx = W_0x + \frac{\alpha}{r}BAx,\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r \ll d$$

- $A$ is initialised randomly (Gaussian), $B$ is initialised to **zero** → $\Delta W = 0$ at step 0, so training starts exactly at the pretrained model.
- Typical: `r = 8–64`, `alpha = 2r`, applied to the attention `q_proj, k_proj, v_proj, o_proj` (and often the MLP projections too).
- **Zero added inference latency** — you can merge $BA$ back into $W_0$ after training.
- **Swappable adapters:** one 7B base model + 50 tiny task adapters (a few MB each).

**QLoRA** = LoRA on a base model quantised to **4-bit NF4**, with double quantisation and paged optimizers. Enables fine-tuning a 65B model on a single 48GB GPU.

**Other PEFT methods:** Adapters (bottleneck layers inserted in each block), Prefix/Prompt tuning (learn virtual token embeddings), IA³ (learn per-channel rescaling vectors), DoRA (decompose magnitude + direction).

```python
from peft import LoraConfig, get_peft_model
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                 task_type="CAUSAL_LM",
                 target_modules=["q_proj","k_proj","v_proj","o_proj"])
model = get_peft_model(base_model, cfg)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062
```

### 14.3 Catastrophic forgetting
Fine-tuning hard on a narrow task degrades general ability. Mitigations: low learning rate (**1e-5 to 5e-5** for full FT; **1e-4 to 3e-4** for LoRA), fewer epochs (2–4), **mix in replay data** from the general distribution, PEFT (frozen base is inherently protective), and layer-wise LR decay.

---
## 15. The LLM Era: Decoding, Prompting, RAG, Agents, Alignment

### 15.1 What "large" bought us

- **Scaling laws** (Kaplan 2020; **Chinchilla**, Hoffmann 2022): loss falls predictably with compute, data, and parameters. Chinchilla's correction — most models were **badly undertrained**; the compute-optimal ratio is roughly **20 tokens per parameter**. Post-2023 practice deliberately *overtrains* small models past compute-optimality because **inference cost dominates** over a model's lifetime.
- **Emergent in-context learning:** GPT-3 showed that few-shot examples in the prompt substitute for gradient updates. The model isn't learning weights — it's pattern-matching the demonstrated task.
- **The 2024–26 shift:** from *train-time* scaling to **test-time compute scaling** — let the model think longer (long chains of thought, sampling + verification) instead of only making it bigger.

### 15.2 Decoding strategies ⭐ (very commonly asked, often poorly answered)

| Strategy | How | Use for |
|---|---|---|
| **Greedy** | always argmax | deterministic, but repetitive & bland |
| **Beam search** (k=4–8) | keep k best partial sequences by total log-prob | **translation, summarization** — tasks with one right answer |
| **Temperature** $p_i \propto \exp(z_i/T)$ | T<1 sharpens, T>1 flattens, T→0 = greedy | global creativity dial |
| **Top-k** | sample from the k most likely tokens | k is fixed, so it's wrong when the distribution is very peaked *or* very flat |
| **Top-p / nucleus** | sample from the smallest set whose cumulative prob ≥ p (0.9–0.95) | ✅ **the default for open-ended text** — adapts set size to the distribution |
| **Min-p** | keep tokens with prob ≥ `min_p × p_max` | newer, robust at high temperature |
| **Repetition / frequency / presence penalty** | downweight already-used tokens | kills loops |
| **Speculative decoding** | a small draft model proposes k tokens; the big model verifies in one pass | **2–3× faster inference, output distribution unchanged** |

> 🪤 **"Why isn't beam search used for chat?"** Beam search maximises sequence likelihood, and the highest-likelihood text is **generic, repetitive and degenerate** ("I don't know. I don't know."). Human-like text is high-entropy — it deliberately sits below the mode. Hence sampling for open-ended generation, beam search for constrained transduction.

### 15.3 Prompting techniques

| Technique | Core idea |
|---|---|
| **Zero-shot** | instruction only |
| **Few-shot** | 3–8 demonstrations; **format consistency matters more than example correctness** |
| **Chain-of-Thought (CoT)** | "think step by step" → externalises reasoning into tokens; works because computation is proportional to tokens generated. Mainly helps at scale. |
| **Self-consistency** | sample N CoT paths, **majority-vote** the final answers |
| **ReAct** | interleave *Reason* and *Act* (tool call) steps — the foundation of agents |
| **Tree/Graph of Thought** | explore and prune multiple branches |
| **Structured output** | JSON schema / constrained decoding / grammars — the production way to get parseable output |
| **Prompt chaining** | decompose into small, individually-testable calls |

**Prompt-engineering hygiene:** be specific; put instructions *and* long context in a clear order; use delimiters (XML tags work well); give the model an out ("say UNKNOWN if not in the context"); specify the output format exactly; iterate against an eval set, not vibes.

### 15.4 RAG (Retrieval-Augmented Generation)

**The problem it solves:** LLMs have a frozen knowledge cutoff, hallucinate confidently, and can't cite. RAG grounds generation in retrieved evidence — updatable without retraining, and attributable.

```text
INDEXING (offline)
  documents → chunk → embed → vector DB (+ BM25 index)

QUERY TIME (online)
  query → [rewrite / expand] → ┬─ dense retrieval (embeddings)  ─┐
                               └─ sparse retrieval (BM25)       ─┴→ fuse (RRF)
        → cross-encoder rerank → top-k context
        → prompt: [instruction + context + query] → LLM → answer + citations
```

**Where RAG actually breaks (this is what senior interviews probe):**

| Failure | Fix |
|---|---|
| Bad chunking splits an answer in half | semantic / recursive chunking, **overlap** (10–20%), respect document structure; **parent-document retrieval** (embed small, return big) |
| Embedding misses exact keywords, IDs, error codes | **hybrid search** — BM25 + dense fused with Reciprocal Rank Fusion |
| Right doc retrieved but ranked #40 | **cross-encoder reranker** |
| Vague/underspecified query | query rewriting, multi-query expansion, **HyDE** (generate a hypothetical answer, embed *that*) |
| "Lost in the middle" — model ignores mid-context evidence | put the most relevant chunk first *and* last; keep k small |
| Model answers from parametric memory anyway | instruct "answer only from context"; require quoted citations; verify |

**Chunk-size rule of thumb:** 256–512 tokens with ~50-token overlap for QA; larger for narrative/summarization.

**RAG evaluation** — decompose it, don't score end-to-end only:
- **Retrieval:** Recall@k, MRR, nDCG, context precision.
- **Generation:** faithfulness/groundedness (is every claim supported?), answer relevance.
- Frameworks: RAGAS, TruLens, DeepEval.

**Vector stores:** FAISS (library, local), Chroma (prototyping), Qdrant / Weaviate / Milvus (production), pgvector (already have Postgres → start here), Pinecone (managed). Index types: **HNSW** (graph, best recall/latency), IVF-PQ (compressed, memory-efficient). All are **approximate** nearest neighbour — you are trading recall for speed, and you should measure that trade-off.

**2026 embedding-model note:** open-weight models now match or beat closed ones on retrieval, **Matryoshka (MRL)** training lets you truncate 3072-d → 1024-d for ~3× storage savings at ~2% quality cost, and **the MTEB leaderboard is a prior, not an oracle** — models routinely reorder on in-domain evals. Always benchmark on ~100–200 of *your* queries before committing, because re-embedding a large corpus later is expensive.

### 15.5 Alignment: SFT → RLHF → DPO

```text
1. PRETRAIN        next-token prediction on trillions of tokens  → raw capability
2. SFT             fine-tune on curated (instruction, response)  → follows instructions
3. PREFERENCE      humans rank outputs A > B                     → captures taste/safety
   ├─ RLHF: train a reward model on the rankings, then optimise the policy with PPO
   │        (+ a KL penalty to the SFT model so it doesn't drift/reward-hack)
   └─ DPO:  skip the reward model — a closed-form classification loss directly on
            preference pairs. Simpler, more stable, no RL loop. Now very common.
4. (2025–26) RLVR  RL with *Verifiable* Rewards — math/code where correctness is
                   checkable by execution. Powers the reasoning-model wave. Variants: GRPO.
```

**Terms to have ready:** *reward hacking* (policy exploits the reward model's flaws), *KL penalty* (leash to the reference policy), *alignment tax* (capability lost to safety tuning), *Constitutional AI* (AI feedback against written principles instead of per-item human labels).

### 15.6 Agents & tool use (the 2026 core)

An **agent** = LLM + tools + memory + a loop, given a goal rather than a prompt.

```text
    ┌──────────────────────────────────────────┐
    │  Goal → PLAN → ACT (tool) → OBSERVE →     │
    │            ↑                    │         │
    │            └──── REFLECT ◄──────┘         │
    │                    ↓                      │
    │              done → ANSWER                │
    └──────────────────────────────────────────┘
```

- **Function/tool calling:** the model emits a structured call; your runtime executes it and feeds back the result.
- **MCP (Model Context Protocol):** an open standard for connecting models to tools and data sources — by 2026 it's a default integration layer rather than an add-on.
- **Memory:** short-term (context window) vs long-term (vector store / summary buffer).
- **Multi-agent:** planner/worker/critic decompositions; increasingly standardised.
- **Real failure modes to name:** compounding errors over long horizons, infinite loops, context-window exhaustion, cost blowups, and **prompt injection** via retrieved or browsed content.

### 15.7 Hallucination — a structured answer

**Why it happens:** the training objective is *next-token likelihood*, not truth. The model has no calibrated notion of "I don't know," and RLHF can actively reward confident-sounding answers.

| Type | Example | Mitigation |
|---|---|---|
| **Factual** | invented citation or date | RAG + grounded citations |
| **Faithfulness** | contradicts the provided context | instruction constraints, NLI-based verification |
| **Reasoning** | valid-looking but wrong chain | CoT + self-consistency + external tools (calculator, code) |

**Mitigation stack:** ground with retrieval → constrain output format → verify with a second pass or a checker → surface uncertainty (logprobs, ensemble disagreement) → keep a human in the loop for high stakes.

---

## 16. Evaluation Metrics

### 16.1 Classification

| Metric | Formula | Use when |
|---|---|---|
| Accuracy | correct / total | **balanced** classes only |
| Precision | TP/(TP+FP) | false positives are costly (spam filter) |
| Recall | TP/(TP+FN) | false negatives are costly (disease, fraud) |
| **F1** | $2\frac{PR}{P+R}$ | need a single balanced number |
| **Macro-F1** | unweighted mean over classes | **imbalanced data** — treats rare classes equally ⭐ |
| Micro-F1 | pool all TP/FP/FN globally | = accuracy in single-label multi-class |
| Weighted-F1 | mean weighted by support | dominated by majority class |
| ROC-AUC / PR-AUC | ranking quality | **PR-AUC for heavy imbalance** |
| MCC | balanced correlation coefficient | robust single number |

> 🪤 **"99% of emails are ham; my model gets 99% accuracy."** It predicts "ham" always. Report **macro-F1 / PR-AUC**, and inspect the confusion matrix.

### 16.2 Language modelling

$$\text{Perplexity}=\exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(w_i \mid w_{<i})\right)=\exp(\text{cross-entropy})$$

**Interpretation:** the model's effective branching factor — "how many equally-likely words is it choosing between?" PPL 20 ≈ as confused as picking uniformly among 20 words. Lower = better. ⚠️ **Only comparable across models with the same tokenizer and same test set.**

### 16.3 Generation

| Metric | What it measures | Weakness |
|---|---|---|
| **BLEU** | n-gram **precision** vs references + **brevity penalty** | precision-only; hates valid paraphrase; corpus-level |
| **ROUGE-N / ROUGE-L** | n-gram / longest-common-subsequence **recall** | rewards copying; ignores fluency |
| **METEOR** | unigram matching + stems + synonyms, recall-weighted | needs linguistic resources |
| **chrF** | character n-gram F-score | strong for morphologically rich languages |
| **BERTScore** | cosine similarity of contextual embeddings | ✅ captures paraphrase; ❌ opaque, model-dependent |
| **BLEURT / COMET** | learned, human-calibrated metrics | ✅ best correlation with humans (COMET for MT) |
| **LLM-as-a-judge** | a strong model scores against a rubric | ✅ flexible, scales; ❌ **position bias, verbosity bias, self-preference bias** — mitigate by randomising order and using a rubric + few-shot anchors |

> 🎤 **The mature answer to "how do you evaluate a summarizer?"** "ROUGE as a cheap regression guard, BERTScore for semantic overlap, an LLM judge with a rubric for faithfulness/coherence, and a small human-annotated gold set as ground truth — because ROUGE alone happily rewards a copy-paste extractive baseline."

### 16.4 Retrieval

**Recall@k**, **Precision@k**, **MRR** (mean reciprocal rank of the first relevant hit), **nDCG@k** (graded relevance with position discounting — the standard for ranking), **MAP**. For RAG specifically, **Recall@k is usually the metric that matters**: the generator can't use what wasn't retrieved.

### 16.5 Benchmarks worth naming
GLUE / SuperGLUE (classic NLU), SQuAD 2.0 (QA with unanswerables), MMLU / MMLU-Pro (knowledge), GSM8K & MATH (math reasoning), HumanEval / MBPP / SWE-bench (code), HELM, MTEB (embeddings), LMSYS Chatbot Arena (human preference Elo), plus **contamination-resistant / private held-out evals** — increasingly the only trustworthy ones.

---
## 17. Core NLP Tasks

| Task | Input → Output | Standard approach (2026) | Metric |
|---|---|---|---|
| **Text classification** | doc → label | TF-IDF+LinearSVC baseline → fine-tuned encoder → LLM zero-shot | Macro-F1 |
| **Sentiment / ABSA** | text → polarity (per aspect) | fine-tuned encoder; LLM for aspect extraction | F1 |
| **NER** | tokens → entity spans | encoder + token classification head (**BIO tagging**) + CRF | **entity-level** F1 |
| **POS tagging** | tokens → tags | encoder token classification | accuracy |
| **Extractive QA** | (question, passage) → span | predict **start/end** logits over tokens | EM / token-F1 |
| **Abstractive QA / RAG** | question → free text | retrieval + LLM | faithfulness, human eval |
| **Summarization** | doc → summary | extractive (TextRank/LexRank) or abstractive (BART/T5/LLM) | ROUGE + BERTScore + judge |
| **Machine translation** | src → tgt | encoder-decoder / multilingual LLM | BLEU, **chrF**, COMET |
| **NLI** | (premise, hypothesis) → ent/neut/contra | cross-encoder | accuracy |
| **Semantic search** | query → ranked docs | bi-encoder + BM25 + reranker | nDCG, Recall@k |
| **Topic modelling** | corpus → topics | LDA (classic), **BERTopic** (embeddings + UMAP + HDBSCAN + c-TF-IDF) | coherence (NPMI) |
| **Coreference** | text → mention clusters | span-based neural / LLM | CoNLL-F1 |
| **Text-to-SQL / structured extraction** | text → SQL/JSON | LLM + schema in prompt + constrained decoding | execution accuracy |

### BIO tagging — the NER detail people forget

```text
Tokens:  Tim   Cook   visited   New    York   in   April
BIO:     B-PER I-PER  O         B-LOC  I-LOC  O    B-DATE
```
`B-` = beginning of an entity, `I-` = inside, `O` = outside. Variants: BIOES/BILOU (adds End/Single) — slightly better boundary modelling.

> 🪤 **Always score NER at the *entity* level, not the token level.** Getting `B-PER` right but `I-PER` wrong means you extracted the wrong entity, yet token accuracy still looks ~95% because most tokens are `O`. A CRF layer on top helps by enforcing valid transitions (you can't go `O → I-PER`).

### Extractive vs abstractive summarization
- **Extractive** — select and stitch existing sentences (TextRank = PageRank over a sentence-similarity graph). Guaranteed faithful, sometimes incoherent.
- **Abstractive** — generate new text. Fluent, but can hallucinate. In production, a **hybrid** (extract candidate evidence, then abstract over it) reduces hallucination meaningfully.

---

## 18. Production & System Design

### 18.1 The interview-proof system-design template

When asked *"design a support-ticket classifier"* or *"build a document QA system,"* walk this ladder out loud:

1. **Clarify** — volume/QPS, latency SLA, languages, label taxonomy, how much labelled data exists, cost ceiling, accuracy vs coverage, privacy/on-prem constraints.
2. **Data** — sourcing, labelling protocol, **inter-annotator agreement (Cohen's κ)**, class balance, and a **held-out set split by time or by user** (not randomly — random splits leak).
3. **Baseline first** — TF-IDF + linear model. Always. It sets the bar and ships in a day.
4. **Model ladder** — baseline → fine-tuned small encoder → LLM zero/few-shot → LLM fine-tune/distil.
5. **Serving** — batching, ONNX/TensorRT, quantisation (INT8/4-bit), distillation, caching (exact + semantic), vLLM for LLMs.
6. **Monitoring** — latency p50/p95/p99, cost/request, **data drift** (embedding-distribution shift, PSI), **concept drift**, confidence distribution, human-review queue for low-confidence cases.
7. **Feedback loop** — log inputs+outputs+corrections, active learning on uncertain samples, scheduled retraining, shadow deploy → A/B.

> 🎤 **The line that impresses:** "I'd ship the TF-IDF baseline to production in week one to get real traffic and real labels, then earn the right to a Transformer with measured lift."

### 18.2 Class imbalance
Class weights / `class_weight='balanced'` · focal loss · resampling (SMOTE is unreliable for text — interpolating embeddings makes non-sentences; prefer **back-translation, EDA, or LLM-generated paraphrases**) · threshold tuning on the PR curve · treat it as anomaly detection if extreme.

### 18.3 When NOT to use an LLM (a genuinely strong answer)
Sub-50ms latency budgets · millions of cheap, repetitive classifications · strict determinism/auditability · data can't leave the premises and you lack GPUs · the task is genuinely simple. A fine-tuned MiniLM/DistilBERT classifier can hit high accuracy in tens of milliseconds at a fraction of a cent per thousand calls, while an LLM is slower, pricier, **and often less accurate on a narrow well-specified task.** NLP ≠ LLM: **all LLMs are NLP, not all NLP should be LLM.**

### 18.4 Cost & latency levers
Quantisation (INT8/FP8/4-bit) · distillation to a smaller student · **KV-cache reuse and prompt caching** · **speculative decoding** · **routing** (cheap model first, escalate on low confidence) · shorter prompts (tokens = money) · batch offline workloads · semantic caching of repeated queries.

### 18.5 Responsible NLP
**Bias** — embeddings inherit corpus stereotypes (WEAT tests, Bolukbasi's `man:programmer :: woman:homemaker`); audit per-group performance, not just aggregate. **Privacy** — PII scrubbing, memorisation/extraction attacks, differential privacy, on-device inference. **Security** — **prompt injection** (untrusted retrieved/browsed text carrying instructions), jailbreaks, training-data poisoning, indirect exfiltration. **Transparency** — model cards, data statements, citations. **Environment** — training and inference carbon.

---

## 19. 2025–2026 Trends

*(Your answer to "what's changed recently?" — sources listed at the end of this section.)*

**1. Mixture of Experts (MoE) is the mainstream paradigm.** Replace the dense FFN with N experts plus a router that activates only k per token. DeepSeek V3 is the reference point: **671B total parameters, ~37B active**, trained for a reported ~$5.6M. You get large-model quality at small-model inference FLOPs; the costs are memory footprint (all experts must be resident) and routing/load-balancing complexity. Notation like `120B-A12B` means 120B total / 12B active.

**2. Test-time compute scaling.** Reasoning models spend more tokens *thinking* before answering, trained with **RL on Verifiable Rewards (RLVR)** on math/code where correctness can be checked by execution. Quality now scales with the *inference* budget, not only the training budget — a genuine break from the 2020–2023 scaling-law framing.

**3. Hybrid architectures beyond pure attention.** 2026 architecture work has moved past "make the Transformer bigger." Nemotron-3 interleaves **Mamba-2** layers with attention; Qwen3.6 uses **Gated DeltaNet** layers similarly. Motivation: **long-context efficiency is the binding constraint** now that models sit inside agent harnesses consuming ever-longer contexts. SSM layers give O(n) sequence mixing; a few attention layers preserve exact long-range recall.

**4. Agents, MCP and RAG became architecture, not add-ons.** The modular phase is over — in 2026 these are core layers of enterprise systems, with multi-agent collaboration frameworks standardising and MCP acting as the default tool/data integration protocol.

**5. Embeddings went multimodal and dimension-flexible.** Gemini Embedding 2 (March 2026) handles five modalities — text, image, video, audio, PDF — over 100+ languages with native **Matryoshka Representation Learning** and 3072-d output. Jina Embeddings v4 swaps between three LoRA adapters (`retrieval.query` / `retrieval.passage` / `text-matching`). Open-weight models (Qwen3-Embedding, BGE-M3, NV-Embed) now match or beat closed APIs on retrieval, so the choice is largely operational and compliance-driven rather than a quality gap.

**6. Long context is table stakes — but "lost in the middle" persists.** 128k–1M windows via RoPE scaling (YaRN/NTK), sliding/ring attention, and hybrid layers. Long context has **not** killed RAG: retrieval remains cheaper, faster, more current, and citable, and attention still degrades on evidence buried mid-context.

**7. Small and on-device models.** 1–8B models now handle classification, extraction and routing at near-zero cost, driven by privacy expectations and latency budgets. Expect "route to the cheapest model that clears the bar" architectures as standard.

**8. Efficiency and safety over raw scale.** The centre of gravity has shifted from chasing parameter counts to inference efficiency (quantisation, distillation, speculative decoding, KV-cache tricks) and to safety/governance being engineered in rather than bolted on.

**9. Classical NLP is not dead — and saying so scores points.** NLP still dominates high-volume, latency-critical work: a fine-tuned BERT classifying clinical notes can hit ~96% accuracy in ~50ms for well under a tenth of a cent per call — faster, cheaper and often *more accurate* than an LLM on that narrow task. **All LLMs are NLP; not all NLP should be LLM.**

**10. Evaluation is the real bottleneck.** Benchmark contamination, saturated leaderboards, and LLM-judge biases (position, verbosity, self-preference) have pushed the field toward private held-out sets, arena-style human preference (Elo), and task-specific in-domain evals. On embeddings specifically: **MTEB is a useful prior, not a decision oracle** — models routinely reorder on in-domain evaluation sets.

<details>
<summary>📚 Sources for this section</summary>

- Sebastian Raschka, *LLM Research Papers: The 2026 List (Jan–May)* — hybrid architectures, long-context efficiency
- *LLM Research Progress and Trends Report 2026* — MoE mainstreaming, test-time compute, multi-agent standardisation
- Prem AI / Tensoria, *Best Embedding Models for RAG (2026)* — MTEB caveats, Matryoshka, open vs closed
- Cheney Zhang, *Which Embedding Model Should You Actually Use in 2026?* — Gemini Embedding 2, Jina v4
- Gaper, *NLP vs LLM (2026)* — when classical NLP still wins
- DEV, *The Future of NLP: Trends, Techniques and Tools in 2026* — on-device NLP, privacy

</details>

---
## 20. ❓ Interview FAQ

> Format: **Q** → the crisp answer you'd actually say, then the follow-up they'll ask.

### A. Representations & Embeddings

**1. What is the difference between BoW and TF-IDF?**
BoW records raw presence/count. TF-IDF multiplies that count by a global rarity term, so words common across the whole corpus are downweighted and distinctive words are boosted. Both are sparse, order-free, and context-free.

**2. Why can't we just use one-hot encoding?**
Dimensionality equals vocabulary size, and **every pair of distinct words has cosine similarity exactly 0** — so the representation encodes no relationships. It's an index, not a vector space.

**3. What's the curse of dimensionality in NLP?**
Sparse text vectors live in 50k+ dimensions where distances concentrate (everything is roughly equidistant), most cells are zero, and the data needed to cover the space grows exponentially. Dense embeddings compress into 100–1000 informative dimensions where distance is meaningful again.

**4. Explain Word2Vec to a non-technical person.**
"We train a model to guess which words hang out near which other words. To do that well, it has to place words that keep similar company close together on an internal map. We throw away the guessing part and keep the map."

**5. CBOW vs Skip-Gram — which and why?**
CBOW predicts the centre word from averaged context (faster, better on frequent words); Skip-Gram predicts context from the centre word (slower, **much better on rare words and small corpora**). Default to Skip-Gram unless training time is the constraint.

**6. Why negative sampling?**
The full softmax denominator sums over the whole vocabulary per training pair — infeasible. Negative sampling reframes it as binary classification against `k` sampled noise words, turning O(V) into O(k).

**7. Why the 3/4 power in the noise distribution?**
It's a smoothing exponent: it shrinks the sampling probability of ultra-frequent words and boosts rare ones relative to plain unigram sampling, so rare words get enough gradient signal. Found empirically.

**8. Word2Vec vs GloVe?**
Predictive local windows vs. weighted least-squares factorisation of a global co-occurrence matrix. Word2Vec is online and memory-light and can be updated incrementally; GloVe uses global statistics and trains fast on a fixed corpus. Both produce static vectors and share the same two limits.

**9. How does FastText solve OOV?**
A word's vector is the **sum of its character n-gram vectors**, so an unseen string still has known subword pieces. Also a big win for morphologically rich languages.

**10. What are the two fundamental limits of static embeddings?**
**Polysemy** — one fixed vector per string, so `bank` can't be both riverbank and financial institution. **OOV** — unseen tokens fall back to `[UNK]` and adaptation requires full retraining.

**11. How do you evaluate word embeddings?**
*Intrinsic:* word similarity correlation (WordSim-353, SimLex-999), analogy accuracy (Google analogy set), clustering coherence. *Extrinsic (what actually matters):* plug them into your downstream task and measure the delta. Intrinsic scores correlate only loosely with downstream gains — say that.

**12. How many dimensions should embeddings have?**
100–300 for classic word vectors; more capacity needs more data, and returns flatten. For modern embedding models, 384–1024 is typical, and Matryoshka-trained models let you truncate at inference to trade storage for a small quality loss.

**13. Can Word2Vec be updated with new data?**
Yes — `build_vocab(..., update=True)` then continue training. GloVe cannot be updated incrementally; it needs the co-occurrence matrix rebuilt.

**14. How would you embed a whole document?**
Ladder: mean pooling → IDF-weighted mean → SIF (weighted mean + remove first principal component) → Doc2Vec → **SBERT/modern embedding model** (correct answer for similarity). Never raw BERT `[CLS]` without fine-tuning.

**15. Why is vanilla BERT bad for sentence similarity?**
MLM+NSP never optimises for cosine comparability, and the resulting space is **anisotropic** — vectors crowd into a narrow cone so everything scores highly similar. SBERT's siamese contrastive fine-tuning fixes this.

### B. Sequence models & Transformers

**16. Why did RNNs fail on long sequences?**
Vanishing gradients: backprop multiplies the same Jacobian per timestep, so signal decays exponentially. Plus sequential computation prevents parallelism.

**17. How exactly does LSTM fix that?**
The cell state is updated **additively** (`C_t = f⊙C_{t-1} + i⊙C̃_t`), giving gradients a near-multiplication-free path backwards. The forget gate can hold ≈1 to keep memory alive.

**18. LSTM vs GRU?**
GRU merges forget+input into an update gate and drops the separate cell state → ~25% fewer parameters, faster, less overfitting on small data. Performance is broadly comparable; LSTM edges ahead on very long sequences.

**19. What problem did attention originally solve?**
The seq2seq **fixed-vector bottleneck** — compressing an arbitrarily long source into one vector. Attention lets the decoder re-weight all encoder states at every output step.

**20. Explain self-attention in one sentence.**
Every token emits a Query, a Key and a Value; it scores its Query against all Keys, softmaxes those into weights, and outputs a weighted blend of all Values — a soft, differentiable dictionary lookup over the whole sequence.

**21. Why divide by √d_k?**
Dot products of d_k-dimensional unit-variance vectors have variance d_k. Unscaled, the softmax saturates into near-one-hot and gradients vanish. Scaling restores unit variance.

**22. Why multiple heads?**
A single softmax gives one averaged view. Multiple heads attend to different representation subspaces in parallel (syntax, coreference, positional patterns) at roughly the same total compute, since each head is `d_model/h` wide.

**23. Why LayerNorm instead of BatchNorm?**
LayerNorm normalises per token across features — independent of batch size and sequence length. BatchNorm's batch statistics are unstable with variable-length, padded text and break at batch size 1 during inference.

**24. Pre-LN vs Post-LN?**
Post-LN (original) needs LR warmup and is fragile at depth. Pre-LN (`x + Attn(LN(x))`) is far more stable and is the modern default.

**25. Why positional encodings?**
Self-attention is permutation-invariant — it treats the input as a set. Position must be injected explicitly. Modern default is **RoPE**, because it makes the attention dot product depend on *relative* distance and extrapolates to longer contexts with interpolation (YaRN).

**26. What's the complexity of self-attention, and how is it mitigated?**
O(n²·d) time and memory in sequence length. Mitigations: FlashAttention (exact, IO-aware — memory becomes O(n)), sliding-window/sparse attention, linear attention, GQA/MQA for the KV cache, and hybrid SSM-attention layers.

**27. What is the KV cache and why does it matter?**
At generation time, past Keys and Values are cached so each new token costs O(n) instead of recomputing O(n²). It is *the* reason autoregressive inference is tractable — and it's also the dominant inference memory cost, which is why GQA and PagedAttention exist.

**28. How many parameters does BERT-base have, and where are they?**
~110M. Roughly a third in the embedding matrix (30k × 768), the rest in 12 blocks; within each block the **FFN holds about two-thirds** of the parameters.

### C. Pretraining & fine-tuning

**29. BERT vs GPT?**
Encoder with bidirectional attention trained on masked LM (understanding) vs decoder with causal attention trained on next-token prediction (generation). The causal mask is the entire difference in objective terms.

**30. Why mask 15%, and why the 80/10/10 split?**
15% balances signal per example against having enough context left. The split avoids a **pretrain–finetune mismatch**: `[MASK]` never appears at inference, so sometimes leaving the real or a random token forces good representations for all positions.

**31. What did RoBERTa change?**
Removed NSP, switched to dynamic masking, trained on ~10× more data with larger batches and longer schedules, and used byte-level BPE. The lesson: **BERT was undertrained**.

**32. What is ELECTRA's trick?**
Replaced-token detection instead of MLM — a small generator corrupts tokens and the discriminator classifies **every** token as original or replaced. Learning signal from 100% of positions instead of 15% → much better sample efficiency.

**33. Explain knowledge distillation.**
Train a small "student" to match a large "teacher's" **soft probability distribution** (temperature-softened logits), often plus the hard labels and intermediate-layer matching. The soft targets carry "dark knowledge" — the teacher's relative confidence across wrong classes — which is richer supervision than a one-hot label. DistilBERT: 40% smaller, 60% faster, ~97% of performance.

**34. When do you fine-tune vs prompt vs RAG?**
**RAG for knowledge, fine-tuning for behaviour, prompting for everything you can get away with.** Try in that cost order: prompt → few-shot → RAG → PEFT → full fine-tune.

**35. Explain LoRA.**
Freeze W, learn a low-rank update `ΔW = BA` with `r ≪ d`. B is zero-initialised so training starts at the pretrained model. Trains ~0.1–2% of parameters, adds **zero inference latency** once merged, and adapters are swappable per task.

**36. What is QLoRA?**
LoRA on top of a 4-bit NF4-quantised frozen base, plus double quantisation and paged optimizers — makes 65B fine-tuning fit on a single 48GB GPU.

**37. What is catastrophic forgetting and how do you prevent it?**
Narrow fine-tuning erases general ability. Prevent with low LR, few epochs, replay/mixed general data, PEFT (frozen base), and layer-wise LR decay.

**38. What learning rate would you use?**
Full fine-tuning of an encoder: **2e-5 to 5e-5**, linear warmup ~6–10% of steps then decay, 2–4 epochs. LoRA: **1e-4 to 3e-4**. Pretraining from scratch: 1e-4 with cosine schedule.

### D. LLMs, RAG, generation

**39. Temperature vs top-k vs top-p?**
Temperature rescales the whole distribution (creativity dial). Top-k truncates to a **fixed count**. Top-p truncates to the smallest set whose cumulative probability exceeds p — **adaptive**, which is why it's the default for open-ended text.

**40. Why not beam search for chatbots?**
Highest-likelihood text is degenerate — bland and repetitive. Human-like output is deliberately high-entropy. Beam search is right for translation/summarization where a single correct output exists.

**41. What is speculative decoding?**
A small draft model proposes k tokens; the large model verifies them in a single forward pass and accepts the longest valid prefix. 2–3× speedup with a **provably unchanged output distribution**.

**42. Why do LLMs hallucinate?**
The objective is next-token likelihood, not truth; there's no calibrated abstention mechanism; and preference tuning can reward confident-sounding answers. Mitigate with grounding (RAG), constrained output, verification passes, and uncertainty surfacing.

**43. Explain RAG end-to-end.**
Offline: chunk → embed → index (dense + BM25). Online: rewrite query → hybrid retrieve → fuse (RRF) → cross-encoder rerank → assemble prompt with instructions to answer only from context → generate with citations → evaluate retrieval and faithfulness separately.

**44. My RAG returns irrelevant chunks. Debug it.**
Isolate the stage. (a) Is the answer even *in* the corpus? (b) Measure **Recall@k** for retrieval alone. (c) Check chunking — is the answer split across a boundary? (d) Add BM25 for exact terms/IDs. (e) Add a reranker. (f) Try query rewriting/HyDE for vague queries. (g) Only then blame the generator. Never tune the prompt before you've measured retrieval.

**45. Long context (1M tokens) — is RAG obsolete?**
No. Retrieval is cheaper per query, lower latency, updatable without re-stuffing, and gives citations. Long context also degrades on evidence buried mid-window ("lost in the middle"). Best practice is both: retrieve, then use the large window generously.

**46. How do you chunk documents?**
Start at 256–512 tokens with 10–20% overlap; respect structural boundaries (headings, paragraphs, table rows); use recursive/semantic splitting; consider **parent-document retrieval** (embed small chunks, return the enclosing section). Then *measure* — chunking is an empirical hyperparameter, not a doctrine.

**47. Hybrid search — why?**
Dense embeddings capture meaning but blur exact tokens; BM25 nails literal matches (error codes, SKUs, rare names). Fuse the two ranked lists with **Reciprocal Rank Fusion**. Nearly always beats either alone.

**48. RLHF vs DPO?**
RLHF trains a separate reward model on human rankings then optimises the policy with PPO plus a KL penalty. DPO derives a closed-form loss directly on preference pairs — no reward model, no RL loop, more stable and cheaper. RLVR/GRPO extend this to programmatically verifiable rewards.

**49. What is prompt injection and how do you defend?**
Untrusted content (a retrieved doc, a web page, a user file) contains instructions the model follows. Defences: treat retrieved text as **data not instructions** (clear delimiters + system-level precedence), least-privilege tool scopes, human confirmation for irreversible actions, output filtering, and injection-detection classifiers. There is no complete fix — say that honestly.

**50. What is MoE and what does `A12B` mean?**
Mixture of Experts: a router activates k of N expert FFNs per token. `120B-A12B` = 120B total parameters, ~12B **active** per token. Large-model quality at small-model inference FLOPs, at the cost of memory and routing complexity.

### E. Evaluation & practice

**51. Explain perplexity.**
exp(cross-entropy) — the effective branching factor of the model's predictions. Only comparable across models sharing a tokenizer and test set.

**52. BLEU vs ROUGE?**
BLEU is n-gram **precision** + brevity penalty (translation, "don't say wrong things"). ROUGE is **recall**-oriented (summarization, "don't miss things"). ROUGE-L uses longest common subsequence.

**53. What's wrong with BLEU/ROUGE?**
They're surface n-gram overlap: they punish valid paraphrase, reward copying, and correlate weakly with human judgement. Supplement with BERTScore/COMET and an LLM judge with a rubric, anchored by a small human-labelled gold set.

**54. Your classifier has 99% accuracy. Are you happy?**
Only after checking the class balance. On a 99:1 split the majority-class predictor gets 99%. Report macro-F1 and PR-AUC and look at the confusion matrix.

**55. What are the biases of LLM-as-a-judge?**
Position bias (favours the first/last option), verbosity bias (longer = better), self-preference (favours its own family's outputs), and sensitivity to formatting. Mitigate by randomising order, using explicit rubrics with few-shot anchors, and calibrating against human labels.

**56. How do you detect model drift in production?**
Monitor input distribution (embedding-space shift, PSI/KL on features), output distribution (predicted-class mix, confidence histograms), and delayed ground truth where available. Alert on shift; keep a human-review queue for low-confidence predictions; retrain on a schedule *and* on trigger.

**57. How do you split data for text?**
Not randomly if there's structure. Split by **time** (train on past, test on future) for anything with drift, by **document/user/patient** to avoid leakage from near-duplicates, and always deduplicate near-identical texts across splits (MinHash/SimHash) before splitting.

**58. How would you handle 10 labelled examples?**
Zero/few-shot LLM first. Then: LLM-generated synthetic training data + a distilled small model; a frozen embedding model + logistic regression; SetFit (contrastive fine-tuning of a sentence encoder — designed exactly for this); weak supervision/labelling functions; active learning to spend your labelling budget on the most informative examples.

**59. How do you make a Transformer fast enough for 20ms inference?**
Pick a smaller architecture (MiniLM/DistilBERT/ModernBERT-base), distil, quantise to INT8, export to ONNX Runtime or TensorRT, use dynamic batching, cap `max_length` and avoid padding to a fixed 512, cache repeated inputs, and serve on the right hardware. Measure p99, not the mean.

**60. What would you do first on a new text-classification project?**
Look at 100 examples by hand. Then: define the taxonomy and check inter-annotator agreement, build a **TF-IDF + linear baseline**, set up the eval harness and a proper split, and only then reach for a Transformer — with the baseline as the bar it has to beat.

**61. NLP vs LLM — are they the same?**
No. NLP is the whole field, from regexes and TF-IDF to Transformers. LLMs are a recent subset. All LLMs are NLP; plenty of NLP problems should *not* be solved with an LLM — latency, cost, determinism, and privacy constraints frequently favour a small fine-tuned model.

**62. What's something you've read recently?**
Have a real answer ready. Safe, defensible 2025–26 talking points: MoE going mainstream (DeepSeek V3's active-parameter efficiency), test-time compute / RLVR reasoning models, hybrid Mamba-attention architectures for long-context efficiency, Matryoshka embeddings, and MCP standardising tool use.

---
## 21. 🎯 Last-Minute Cheat Sheet

### The one-liners

| Concept | Say this |
|---|---|
| **Distributional hypothesis** | "You shall know a word by the company it keeps." |
| **TF-IDF** | frequent here × rare everywhere = informative |
| **BM25** | TF-IDF + saturation + length normalisation = the real IR baseline |
| **Word2Vec** | fake prediction task, real by-product: the hidden weights |
| **Negative sampling** | turn O(V) softmax into O(k) binary classification |
| **GloVe** | least-squares regression on log co-occurrence counts |
| **FastText** | word = Σ character n-grams ⇒ no OOV |
| **Polysemy bottleneck** | one string → one vector, forever |
| **LSTM** | additive cell state = a gradient highway |
| **Attention** | soft, differentiable dictionary lookup |
| **√d_k** | keeps softmax out of its saturated, zero-gradient regime |
| **Multi-head** | multiple representation subspaces at the same total cost |
| **LayerNorm** | per-token, so batch-size and length independent |
| **Positional encoding** | attention is a set operation; order must be injected |
| **BERT vs GPT** | bidirectional MLM (understand) vs causal LM (generate) |
| **Subword tokenization** | `riverbank → [river, ##bank]` ⇒ no `[UNK]` |
| **LoRA** | ΔW is low-rank; freeze W, learn BA |
| **RAG vs fine-tune** | **knowledge vs behaviour** |
| **Top-p** | adaptive truncation beats fixed-k truncation |
| **Perplexity** | effective branching factor = exp(cross-entropy) |
| **MoE** | activate k of N experts ⇒ big-model quality, small-model FLOPs |

### Numbers worth memorising

| | |
|---|---|
| BERT-base | 12 layers · 768 hidden · 12 heads · 110M params · 512 tokens |
| BERT-large | 24 · 1024 · 16 · 340M |
| MLM masking | 15%, split 80 `[MASK]` / 10 random / 10 unchanged |
| Word2Vec dims | 100–300 · window 5 · min_count 5 · negative 5–20 |
| Negative-sampling exponent | 3/4 |
| GloVe weighting | α = 0.75, x_max = 100 |
| FastText n-grams | n = 3…6 |
| FFN expansion | 4× d_model (≈ ⅔ of block parameters) |
| Chinchilla-optimal | ~20 tokens per parameter |
| English tokenization | ~0.75 words/token (~4 chars/token) |
| Fine-tune LR | 2e-5–5e-5 (full) · 1e-4–3e-4 (LoRA) · 2–4 epochs |
| RAG chunk size | 256–512 tokens, 10–20% overlap |
| DistilBERT | 40% smaller · 60% faster · ~97% performance |

### The 5 questions you're most likely to be asked

1. **"Walk me through how you'd represent text as numbers."** → Give the 5-era arc from §0, then TF-IDF → Word2Vec → BERT with the *why* at each transition.
2. **"Why did Transformers replace RNNs?"** → O(1) sequential ops (parallelism) + O(1) path length (no long-range decay). Cost: O(n²).
3. **"Explain self-attention and why we scale by √d_k."** → Q/K/V soft lookup; scaling prevents softmax saturation.
4. **"RAG or fine-tuning?"** → Knowledge vs behaviour; then describe the full retrieval pipeline and how you'd evaluate each stage separately.
5. **"How would you evaluate this?"** → Never just "accuracy." Name the metric, why it fits the class balance and the cost asymmetry, and what its blind spot is.

### Red flags to avoid saying

- ❌ "TF-IDF is a machine learning model." (It's a deterministic weighting.)
- ❌ "Accuracy is 99%, so it's good." (Check balance.)
- ❌ "I'd just use GPT for everything." (Latency, cost, privacy, determinism.)
- ❌ "BERT generates text." (It's an encoder; it fills masks.)
- ❌ "Cosine and dot product are the same." (Only after L2 normalisation.)
- ❌ "Attention gives us interpretability." (Attention weights are *not* faithful explanations — a well-known caveat, and citing it earns credit.)
- ❌ Fitting the vectorizer on train + test. (Leakage.)

### Closing frame

> **"Every advance in NLP has been about removing a hand-made assumption.**
> BoW assumed order doesn't matter. Word2Vec assumed one meaning per word. RNNs assumed you must read sequentially. Transformers removed all three — and the current frontier is removing the assumption that more parameters is the only way to get more capability."

---

## 📎 Appendix: Further Reading

**Papers (the canon):**
`Efficient Estimation of Word Representations` (Word2Vec, 2013) · `Distributed Representations of Words and Phrases` (negative sampling, 2013) · `GloVe` (2014) · `Enriching Word Vectors with Subword Information` (FastText, 2016) · `Neural MT by Jointly Learning to Align and Translate` (Bahdanau attention, 2014) · **`Attention Is All You Need` (2017)** · `BERT` (2018) · `RoBERTa` (2019) · `Language Models are Few-Shot Learners` (GPT-3, 2020) · `Sentence-BERT` (2019) · `LoRA` (2021) · `Training Compute-Optimal LLMs` (Chinchilla, 2022) · `Chain-of-Thought Prompting` (2022) · `Direct Preference Optimization` (2023) · `FlashAttention` (2022)

**Blogs:** Jay Alammar's *Illustrated Transformer* / *Illustrated Word2Vec* · Lilian Weng's blog · Sebastian Raschka's *Ahead of AI* · Chris McCormick's Word2Vec tutorials · Andrej Karpathy's *Let's build GPT* (build one from scratch — nothing teaches faster)

**Libraries:** `scikit-learn` · `gensim` · `spaCy` · `NLTK` · `transformers` · `sentence-transformers` · `tokenizers` · `datasets` · `peft` · `trl` · `faiss` / `qdrant` / `chroma` · `vllm` · `rank_bm25` · `ragas`

**Practice:** implement Word2Vec skip-gram with negative sampling in NumPy · implement scaled dot-product attention in 10 lines of PyTorch · build a RAG pipeline over your own notes and actually measure Recall@k · fine-tune DistilBERT on a Kaggle text task and beat your own TF-IDF baseline.

---

*Good luck. Explain the **why**, not just the what — that's what separates a hire from a pass.* 🚀
