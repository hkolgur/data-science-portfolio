# NLP Embeddings & Vectorization Interview Cheat Sheet
# NLP Word Representations: Comparative Analysis & Core Mechanics

This note documents how different text embedding techniques process semantic context, using two contrasting sentences containing the homonym **"bank"**:
*   **Sentence A (Geographic):** *"The bank of the river was muddy."*
*   **Sentence B (Financial):** *"The bank approved the cash loan."*

---

## 1. Vocabulary Construction
After lowercasing and removing stop words ("the", "of", "was"), both sentences share a combined vocabulary of **7 unique tokens** arranged alphabetically:
`['approved', 'bank', 'cash', 'loan', 'muddy', 'river', 'string']` -> *Note: 'string' is omitted here for strict vocabulary mapping.*

**Alphabetical Vocabulary Key:**
`[0: approved, 1: bank, 2: cash, 3: loan, 4: muddy, 5: river]`

---

## 2. Binary Bag-of-Words (BOW)
Tracks only the **presence (1)** or **absence (0)** of a token. It completely discards frequency, sequence, and grammar.

### Vector Representations
*   **Sentence A Vector:** `[0, 1, 0, 0, 1, 1]` 
*   **Sentence B Vector:** `[1, 1, 1, 1, 0, 0]`

### ❌ Fatal Flaw for Git Notes
*   **Context Blindness:** Look at index `1` (the word `bank`). Both vectors contain a `1`. 
*   **False Similarity:** Because BOW treats every token independently, a cosine similarity calculation will falsely indicate these sentences are structurally related through `bank`, completely missing that a "river bank" and a "financial bank" are entirely different concepts.

---

## 3. TF-IDF (Term Frequency-Inverse Document Frequency)
Weights tokens by multiplying how often they appear in a document (**TF**) against how rare they are across the entire dataset (**IDF**). 

### Mathematical Intuition
$$\text{TF-IDF} = \text{TF} \times \log\left(\frac{\text{Total Documents}}{\text{Documents containing word}}\right)$$

### Mock Vector Representations
*Assumed Corpus Behavior: "bank" appears in both sentences (Common $\rightarrow$ Low IDF = 0.15). Specialized words like "muddy" or "loan" only appear in one sentence each (Rare $\rightarrow$ High IDF = 1.2).*

*   **Sentence A Vector:** `[0.0, 0.15, 0.0, 0.0, 1.2, 1.2]` *(Order: approved, bank, cash, loan, muddy, river)*
*   **Sentence B Vector:** `[1.2, 0.15, 1.2, 1.2, 0.0, 0.0]`

### ⚠️ Evaluation for Git Notes
*   **Improvement:** It successfully minimizes the impact of the shared word `bank` because it occurs in both contexts. It forces the model to focus on the unique keywords (`river` vs `loan`).
*   **Remaining Flaw:** It still cannot assign different semantic meanings to the word `bank`. The token `bank` is still tied to a single index, forcing it to have the same core value.

---

## 4. Word2Vec (Continuous Bag-of-Words / Skip-Gram)
Generates dense, low-dimensional vectors (e.g., 100-300 dimensions) trained step-by-step using a local sliding context window. Sentence representations are built by averaging individual word vectors.

### Mechanics & Local Optimization
Word2Vec learns by predicting a target word from its immediate neighbors (or vice versa).

*   **Sentence A Processing:** The sliding window binds `bank` strongly to `river` and `muddy`.
*   **Sentence B Processing:** The sliding window binds `bank` strongly to `approved` and `loan`.

### Mock 3D Vectors (Individual vs. Sentence Average)
*   $\vec{v}_{\text{river}} = [0.12, 0.89, -0.45]$
*   $\vec{v}_{\text{loan}} = [-0.78, -0.23, 0.91]$
*   $\vec{v}_{\text{bank}} = [-0.15, 0.22, 0.11]$ *(A single shifting vector)*

$$\vec{v}_{\text{Sentence A (Avg)}} = [0.08, 0.45, -0.21] \quad \longleftrightarrow \quad \vec{v}_{\text{Sentence B (Avg)}} = [-0.41, -0.05, 0.48]$$

### ❌ Fatal Flaw for Git Notes
*   **The Overwrite Problem:** Word2Vec updates weights incrementally via online gradient descent. If your training dataset suddenly feeds Word2Vec 10,000 sentences about financial loans, the single vector for `bank` will be drastically pulled toward financial coordinates, effectively "erasing" or distorting its geographic relationship to rivers.

---

## 5. GloVe (Global Vectors for Word Representation)
Generates dense vectors by performing global matrix factorization on an entire corpus's log-co-occurrence matrix. 

### Mechanics & Global Optimization
GloVe bypasses step-by-step streaming. It constructs a massive global table tracking how often every word pairs with every other word across the whole corpus before training even begins.

Instead of a single window pulling the vector back and forth, GloVe’s loss function forces the vector for `bank` to find a static, globally optimized spatial position that simultaneously balances its ratios with *all* its co-occurring clusters.

### Mock 3D Vectors
*   $\vec{v}_{\text{bank}} = [-0.31, 0.28, 0.25]$ *(Mathematically balanced position)*

###  Key Advantage for Git Notes
*   **Subspace Stability:** While GloVe still yields a single static vector for `bank` (meaning it cannot dynamically change per sentence like modern Transformers), its position is mathematically robust. The vector is constrained by global co-occurrence ratios, preventing it from being radically overwritten or skewed by a sudden local cluster of training data. 
*   **Frequent Word Safety:** High-frequency pairings (e.g., `the` + `bank`) are systematically capped by GloVe's weighting function, ensuring meaningful relationships (`bank` + `river` and `bank` + `loan`) drive the final embedding layout.


## 📊 1. Evolutionary Roadmap: BoW vs. TF-IDF vs. Word Embeddings

### 01. Binary Bag of Words (BoW)
* **Concept:** Represents text based on word presence/absence or raw frequencies.
* **Format:** Highly dense vocabulary lookup resulting in a **sparse, high-dimensional matrix** populated mostly with zeros.
* **Flaw:** Disregards word order, context, and semantic meaning completely.

### 02. TF-IDF (Term Frequency-Inverse Document Frequency)
* **Concept:** A frequency-based representation that scales a term's weight based on how unique it is across the entire corpus.
* **Core Logic:** Considers a term's local importance relative to global document frequencies. Valuable for keyword extraction and classic search tasks.
* **Flaw:** Still completely blind to context or underlying semantic similarities between different words.

### 03. Word Embeddings (Word2Vec / GloVe)
* **Concept:** Low-dimensional, highly dense, continuous mathematical spaces (latent embeddings).
* **Core Logic:** Captures semantic meaning, context, and structural analogies ($King - Man + Woman = Queen$).

---

## ⚖️ 2. Word2Vec: CBOW vs. Skip-gram

CBOW Architecture:     [Context Words]  ───►  [Predict Target Word]

Skip-gram Architecture: [Target Word]   ───►  [Predict Context Words]

| Feature | Continuous Bag of Words (CBOW) | Skip-gram |
| :--- | :--- | :--- |
| **Objective** | Predicts 1 **target** word from multiple **context** words. | Predicts multiple **context** words from 1 **target** word. |
| **Gensim Flag** | `sg=0` (Default) | `sg=1` |
| **Speed** | Faster (\(O(V)\) text processing). | Slower (\(O(C \times V)\) where \(C\) is window size). |
| **Data Fit** | Better for small, dense datasets with clear context. | Requires larger datasets to stabilize gradients safely. |
| **Rare Words** | Smoothes them out; represents rare words poorly. | Excellent at capturing rare words and distinct relationships. |
| **Vector Nature** | Learns the **average semantic theme** of a context window. | Learns **direct, strong pairings** between specific terms. |

### ⚠️ The Training vs. Inference Interview Trap
* **Question:** *"How does `model.wv['word']` run a forward pass prediction?"*
* **Answer:** It doesn't. Training is a semi-supervised process where weights adjust using backpropagation to learn relationships from random initial values. Once trained, the prediction layer is discarded. `model.wv['word']` is a static matrix lookup (\(O(1)\) time complexity).

---

## 🚀 3. Word2Vec Optimization & Math Mechanics

### Solving the Softmax Bottleneck
Computing standard Categorical Cross-Entropy Loss over a massive Vocabulary ($V$) is computationally expensive ($O(|V|)$), cutting final layer complexity down to $O(\log_2 |V|)$. Word2Vec solves this via:
1. **Hierarchical Softmax (`hs=1`):** Uses a binary Huffman tree to evaluate probabilities, cutting final layer complexity down to $O(\log_2 |V|)$).
2. 
3. **Negative Sampling (`negative > 0`):** Instead of calculating updates for all vocabulary elements, it updates weights only for the true target word and a tiny random sample (5–20) of incorrect ("negative") words. This turns a massive multi-class challenge into fast binary logistic regression.

### Subsampling Frequent Words
High-frequency words (*"the"*, *"is"*) crowd out learning opportunities for rare tokens. Word2Vec dynamically drops them using a probability formula based on word frequency:
$$P(w) = 1 - \sqrt{\frac{\text{threshold}}{\text{freq}(w)}}$$

---

## 🌐 4. GloVe (Global Vectors for Word Representation)

### Core Mechanics
* **Global Context:** Unlike Word2Vec which uses local sliding windows, GloVe derives semantic meaning by building a global **word-word co-occurrence matrix** across the entire corpus.
* **Simplicity:** The training pipeline is mathematically simpler and highly parallelizable compared to Word2Vec.

### Key Key Advantages over Word2Vec
* **Frequent Word Safety:** Because it factors in the entire co-occurrence matrix systematically, it natively controls for high frequencies without over-weighting common words the way Skip-gram can.
* **Linear Relationships:** Yields much cleaner, more interpretable linear dimensions for vector arithmetic and analogies.
* **Speed:** Training is substantially faster across massive corpora.

### 🔴 Shared Limitations (Word2Vec & GloVe)
* **Static Nature:** Both generate fixed vectors. They cannot handle polysemy (e.g., *"Apple"* the fruit vs. *"Apple"* the tech company receive the same representation).
* **Out-of-Vocabulary (OOV):** Completely blind to unseen tokens, structural misspellings, or rare vocabulary modifications unless retrained.

---
# GloVe vs. Word2Vec: Handling High-Frequency Words

To see the difference, imagine a small corpus with two sentences:"

The chef made the soup."

"The chef served the steak."

Flooded Training: The model encounters (chef, the), (made, the), (soup, the), (served, the) over and over again.Overwritten

Updates: Word2Vec updates word vectors incrementally through gradient descent. Because "the" appears in almost every window, its vector gets updated constantly.

Drowned Meanings: The heavy gradient updates from "the" drown out the rare, meaningful connections, like (chef, soup) or (chef, steak).

## ❌ Word2Vec (Sliding Window Failure)
* **Incremental Updates:** Slides a local context window step-by-step across the text.
* **Flooded Training:** High-frequency stop words (e.g., "the", "is") generate massive numbers of repetitive training pairs.
* **Drowned Meanings:** Constant gradient updates from common words drown out rare, meaningful semantic connections (e.g., "chef" + "soup").

##  GloVe (Global Matrix Control)
* **Global Counting:** Compresses repetitions into a single global co-occurrence matrix before training.
* **Capped Weighting:** Applies a weighting function $f(X_{ij}) = \min\left(\left(\frac{X_{ij}}{x_{max}}\right)^\alpha, 1.0\right)$ to the raw counts.

  
* **Neutralized Power:** Caps the maximum weight at `1.0` once counts hit a threshold ($x_{max}$), preventing massive frequencies from scaling infinitely.The pair (chef, soup) might get a weight of 0.2. Because 1.0 is not much larger than 0.2, the high-frequency word loses its ability to dominate the mathematical updates.
  
* **Balanced Learning:** Ensures common pairs cannot mathematically dominate the loss function, leaving room for rare words to influence the vectors.
