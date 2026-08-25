# Word2vec Interview Cheat Sheet

## 📌 Core Concept: Input & Architectures
* **Raw Input:** An iterable of tokenized sentences (lists of strings, e.g., `[["word1", "word2"]]`).
* **Mathematical Input:** One-hot encoded vectors processed via a sliding context window.
* **The `sg` Parameter:** Controls the architecture in Gensim (`sg=0` for CBOW, `sg=1` for Skip-gram).

---

## ⚖️ CBOW vs. Skip-gram

| Feature | Continuous Bag of Words (CBOW) | Skip-gram |
| :--- | :--- | :--- |
| **Objective** | Predicts 1 **target** word from multiple **context** words. | Predicts multiple **context** words from 1 **target** word. |
| **Gensim Flag** | `sg=0` (Default) | `sg=1` |
| **Speed** | Faster ($O(V)$ text processing). | Slower ($O(C \times V)$ where C is window size). |
| **Data Fit** | Better for small/dense datasets with clear context. | Requires larger datasets to stabilize gradients. |
| **Rare Words** | Smoothes them out; treats rare words poorly. | Excellent at capturing rare words and distinct relationships. |
| **Vector Nature** | Represents the **average semantic theme** of a context. | Represents **direct, strong pairings** between specific words. |

---

## 🛠️ Vital Hyperparameters
* **`vector_size`**: Dimensionality of the embedding space (typically 100–300). Higher captures more nuance but risks overfitting.
* **`window`**: Maximum distance between target and context word. Large windows capture topical/domain similarity; small windows capture strict grammatical/functional similarity.
* **`min_count`**: Ignores all words with total frequency lower than this. Crucial for filtering out noise/typos.
* **`workers`**: Number of CPU threads used for parallelization (speeds up training).

---

## ⚠️ Common Interview Trap: Training vs. Inference
* **The Trap:** An interviewer might ask, *"How does `model.wv['word']` make a prediction?"*
* **The Reality:** **It doesn't.** Training is purely an optimization phase to learn weights. Once trained, the prediction layer is discarded. `model.wv['word']` is a static matrix lookup ($O(1)$ time complexity) yielding a fixed vector array.

---

## 🚀 Advanced Optimization Details

### 1. Avoiding Performance Bottlenecks
Standard Softmax calculation over a massive vocabulary $V$ is computationally impossible ($O(|V|)$). Word2vec uses two optimization tricks:
* **Hierarchical Softmax (`hs=1`):** Uses a binary Huffman tree to reduce final layer complexity from $O(|V|)$ to $O(\log_2 |V|)$.
* **Negative Sampling (`negative>0`):** Updates the correct word and a small sample (e.g., 5–20) of random "negative" words. This shifts the task from multiclass classification to binary logistic regression.

### 2. Subsampling Frequent Words
* High-frequency words (like *"the"*, *"is"*) don't provide much context. Word2vec uses a parameter `sample` to randomly discard frequent words during training, speeding up the process and improving rare word vectors.
