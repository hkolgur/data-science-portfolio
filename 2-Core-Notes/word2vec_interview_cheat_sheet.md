# NLP Embeddings & Vectorization Interview Cheat Sheet

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
