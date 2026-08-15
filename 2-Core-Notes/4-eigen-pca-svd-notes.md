# Eigenvalues, Eigenvectors, PCA & SVD
### Study notes + interview question bank for mid-level Data Science interviews

> Compiled from two sources: a set of personal notes on dimensionality reduction / recommender systems, and an personal deck on PCA & SVD. **Several numerical errors in the source deck are corrected here** — see Part 9, which is worth reading before you revise from the original slides.

---

## How to use this

Work through Parts 1 → 5 in order; the concepts stack strictly. Parts 6 → 8 are application and breadth (recommenders, non-linear methods, choosing between everything). Part 9 is the question bank, Part 11 is the pre-interview cheat sheet.

**The core insight to hold onto throughout:**

> PCA is eigendecomposition of the covariance matrix. SVD is a more general, more numerically stable way to get the same answer without ever forming the covariance matrix. Everything else is detail.

---

## Part 0 — What interviewers actually test

Mid-level interviews rarely ask you to hand-compute a 4×4 eigendecomposition. They probe six things, roughly in this order of frequency:

| # | What they ask | What they're checking |
|---|---|---|
| 1 | "Explain PCA to me." | Can you give the intuition *and* the mechanics, without waffling? |
| 2 | **"Do you need to scale before PCA? Why?"** | **The single most common follow-up. Get this wrong and the interview turns.** |
| 3 | "How do you choose the number of components?" | Do you know it's a business/validation decision, not a formula? |
| 4 | "What's the relationship between PCA and SVD?" | Depth. Separates memorisers from people who understand it. |
| 5 | "When would PCA *not* work / when would you not use it?" | Judgement. Do you know the assumptions? |
| 6 | "You've reduced to 5 components — what do they mean?" | Do you understand that interpretability is usually lost? |
| 7 | **"Why eigenvectors? Where does that come from?"** | Depth. Very few candidates can derive it — §4.9 takes 90 seconds |
| 8 | "Walk me through PCA on a small example." | Can you actually compute it? Have §4.4 memorised |
| 9 | **"PCA vs t-SNE?" / "can I conclude there are N clusters?"** | **Do you know t-SNE plots are not evidence? §7.3** |

A clean answer to #1 in 60 seconds, plus a confident #2, gets you most of the credit available.

---

# Part 1 — Linear algebra foundations

## 1.1 The objects

| Object | Definition | Example |
|---|---|---|
| **Scalar** | A single number | `10` |
| **Vector** | 1-D array; a point or a direction in space | `[1, 2, 3]` |
| **Matrix** | 2-D array of numbers, rows × columns | `[[1,2],[3,4]]` |
| **Square matrix** | Equal rows and columns | 3×3 |
| **Symmetric matrix** | `A = Aᵀ` — mirror across the diagonal | every covariance matrix |
| **Diagonal matrix** | Non-zero only on the diagonal | eigenvalue matrix `Λ` |
| **Identity `I`** | 1s on the diagonal, 0s elsewhere | acts like "1" for matrices |
| **Orthogonal matrix** | `QᵀQ = I`; columns are unit length and mutually perpendicular | rotation matrices; `U` and `V` in SVD |

## 1.2 A matrix is a transformation

This is the mental model that makes eigenvectors click. **Multiplying a vector by a matrix moves that vector** — it rotates, stretches, squashes, or flips it.

```
v = [1, 0]        A = [[2, 0],
                       [0, 3]]

Av = [2, 0]       → stretched 2× along x, direction unchanged
```

Some matrices rotate vectors, some scale them, most do a mix of both.

## 1.3 Determinant

The determinant is a **scalar that measures how much a matrix scales area/volume**, and — the part that matters here — whether the matrix is invertible.

For 2×2:
```
A = [[a, b],          det(A) = ad − bc
     [c, d]]
```

For 3×3, expand along the first row (cofactor expansion):
```
det(A) = a₁(b₂c₃ − b₃c₂) − a₂(b₁c₃ − b₃c₁) + a₃(b₁c₂ − b₂c₁)
```

**The one fact you must retain:**

> `det(A) = 0` ⟺ A is **singular** ⟺ A is **not invertible** ⟺ A squashes space into a lower dimension ⟺ A has a non-zero vector in its null space (`Av = 0` for some `v ≠ 0`).

That last equivalence is exactly what makes the eigenvalue derivation work in Part 2.

## 1.4 Rank

**Rank = the number of linearly independent rows (or columns).** Informally: how many genuinely distinct directions of information the matrix contains.

- Full rank (rank = min(rows, cols)) → no redundancy.
- Rank deficient → some columns are linear combinations of others → **multicollinearity**.

Rank is the bridge from linear algebra to "why do we need dimensionality reduction": if your 24-feature dataset has effective rank 6, you have 24 columns carrying 6 columns' worth of information.

---

# Part 2 — Eigenvalues & Eigenvectors

## 2.1 The definition

> **An eigenvector of a matrix `A` is a non-zero vector whose direction is unchanged when `A` is applied to it. The eigenvalue is the factor by which it is stretched or shrunk.**

```
A v = λ v
```
- `A` — an `n×n` **square** matrix (eigenvectors are only defined for square matrices). Non-singular/Singular. If singular(det(A) = 0) , at least one of Eigen value will be 0. **A is not a covariance matrix in this** 
- `v` — the eigenvector (direction)
- `λ` — the eigenvalue (magnitude of stretch)

## 2.2 Geometric intuition

Picture a rubber sheet with arrows drawn on it in every direction. Apply the transformation `A` — the sheet stretches and rotates, and most arrows swing to point somewhere new.

**A few special arrows don't swing at all.** They stay on their original line; they just get longer or shorter. Those are the **eigenvectors**, and how much they grow is the **eigenvalue**.

- `λ > 1` → stretched
- `0 < λ < 1` → shrunk
- `λ < 0` → flipped to point the opposite way along the same line
- `λ = 0` → collapsed to the origin (⟹ matrix is singular)

> **Interview soundbite:** *"Eigenvectors are the directions a transformation leaves alone; eigenvalues are how much it stretches them."*

## 2.3 Deriving the characteristic equation

This derivation appears on whiteboards. Know it.

```
Start:              A v = λ v
Move to one side:   A v − λ v = 0
Insert identity:    A v − λ I v = 0      (needed so we can factor — λ is a scalar, A is a matrix)
Factor:             (A − λ I) v = 0
```

⚠️ **Note the order: `(A − λI)v = 0`, not `v(A − λI) = 0`.** The source deck has this backwards. With `v` as a column vector, the dimensions only work on the left.

Now the key step. We want a **non-zero** `v`. If `(A − λI)` were invertible, we could multiply both sides by its inverse and get `v = 0` — the trivial solution we don't want. So for an interesting solution to exist, `(A − λI)` **must be singular**. And from §1.3, singular means determinant zero:

```
det(A − λI) = 0          ← the characteristic equation
```

Solve it for `λ`, and you have the eigenvalues.

## 2.4 Fully worked example

Find the eigenvalues and eigenvectors of:
```
A = [[1, 4],
     [3, 2]]
```

**Step 1 — form `A − λI`:**
```
A − λI = [[1−λ,   4 ],
          [ 3,  2−λ]]
```

**Step 2 — set the determinant to zero:**
```
(1−λ)(2−λ) − (4)(3) = 0
2 − λ − 2λ + λ² − 12 = 0
λ² − 3λ − 10 = 0
(λ − 5)(λ + 2) = 0

λ₁ = 5,  λ₂ = −2
```

**Step 3 — find the eigenvector for λ₁ = 5.** Substitute back and solve `(A − 5I)v = 0`:
```
[[1−5,  4 ],  [x]   [0]        [[−4,  4],  [x]   [0]
 [ 3, 2−5]] × [y] = [0]   →     [ 3, −3]] × [y] = [0]

Row 1:  −4x + 4y = 0  →  y = x
```
Both rows give the same equation — they must, because the matrix is singular by construction. Any vector on the line `y = x` works:
```
v₁ = k·[1, 1]ᵀ ,  k ∈ ℝ
```

**Step 4 — eigenvector for λ₂ = −2.** Solve `(A + 2I)v = 0`:
```
[[3, 4],  [x]   [0]
 [3, 4]] ×[y] = [0]

Row 1:  3x + 4y = 0  →  y = −(3/4)x
```
```
v₂ = k·[4, −3]ᵀ
```

**Verify** (always do this if you have time):
```
A v₁ = [[1,4],[3,2]] [1,1]ᵀ = [5, 5]ᵀ = 5·[1,1]ᵀ  ✓
A v₂ = [[1,4],[3,2]] [4,−3]ᵀ = [−8, 6]ᵀ = −2·[4,−3]ᵀ  ✓
```

```python
import numpy as np
A = np.array([[1, 4], [3, 2]])
w, v = np.linalg.eig(A)
# w -> [-2., 5.]
# v -> columns are UNIT-NORMALISED eigenvectors:
#      [-0.8, 0.6] for λ=-2  (which is [4,-3]/5)
#      [-0.707, -0.707] for λ=5  (which is [1,1]/√2, sign-flipped)
```

> ⚠️ **Eigenvectors are not unique.** Any scalar multiple is also an eigenvector — including the negative. NumPy returns unit-length vectors with an arbitrary sign convention. **This is why PC signs can flip between runs or libraries, and it means nothing.** If an interviewer shows you `[0.84, 0.54]` and `[−0.84, −0.54]` and asks which is right: both are.

## 2.5 Properties worth memorising

| Property | Statement | Why it matters |
|---|---|---|
| Count | An `n×n` matrix has `n` eigenvalues (counting multiplicity, allowing complex) | Sanity check |
| **Trace** | `Σλᵢ = trace(A)` = sum of diagonal | Fast check. Above: 5 + (−2) = 3 = 1 + 2 ✓ |
| **Determinant** | `Πλᵢ = det(A)` | Above: 5 × (−2) = −10 = (1)(2)−(4)(3) ✓ |
| Singularity | `A` singular ⟺ some `λ = 0` | Links back to rank |
| **Symmetric matrices** | All eigenvalues **real**; eigenvectors **mutually orthogonal** | **This is why PCA works and why PCs are uncorrelated** |
| Positive semi-definite | All `λ ≥ 0` | Covariance matrices are always PSD → variance is never negative |
| Rank | rank = number of non-zero eigenvalues | Detects redundancy |

**The symmetric-matrix row is the load-bearing one.** A covariance matrix is symmetric by construction (`Cov(X,Y) = Cov(Y,X)`), which guarantees PCA gives you real, orthogonal axes. If covariance matrices weren't symmetric, PCA wouldn't exist in the form we use it.

## 2.6 Where eigen-decomposition shows up

- **PCA** — eigenvectors of the covariance matrix are the principal components.
- **Eigenfaces** — face recognition; eigenvectors of a face-image covariance matrix look like ghostly "template faces", and any face is approximated as a weighted sum of ~20–100 of them instead of 16,384 raw pixels.
- **PageRank** — Google's original algorithm finds the principal eigenvector of the web's link matrix.
- **Spectral clustering** — eigenvectors of a graph Laplacian.
- **Stability analysis** — in dynamical systems, eigenvalue signs determine whether a system converges or blows up.
- **Regularisation / conditioning** — the *condition number* (ratio of largest to smallest eigenvalue) tells you whether a regression will be numerically unstable. Ridge regression adds `λI`, which shifts every eigenvalue up and away from zero.

---

# Part 3 — Variance, Covariance & the Covariance Matrix

## 3.1 From variance to covariance

**Variance** — how much one variable spreads around its own mean:
```
Var(X) = (1/(n−1)) Σ (xᵢ − x̄)²
```

**Covariance** — how two variables move *together*:
```
Cov(X, Y) = (1/(n−1)) Σ (xᵢ − x̄)(yᵢ − ȳ)
```

Note `Cov(X, X) = Var(X)`. Variance is just covariance with itself — which is why they share one matrix.

| Sign | Meaning |
|---|---|
| `Cov > 0` | When X is above its mean, Y tends to be too |
| `Cov < 0` | When X is above its mean, Y tends to be below |
| `Cov ≈ 0` | No **linear** relationship (they can still be strongly related non-linearly) |

## 3.2 `n` or `n−1`?

- **`n−1` (sample covariance)** — the default in `numpy.cov`, `pandas.cov`, and virtually all statistics software. Corrects for the fact that you estimated the mean from the same data (Bessel's correction), giving an unbiased estimate.
- **`n` (population covariance)** — used when you truly have the whole population, and internally in some ML libraries.

For PCA it makes **no practical difference to the components** — it scales every eigenvalue by the same constant, so directions and explained-variance *ratios* are unchanged. But say `n−1` if asked; it's the statistically correct default. *(The source deck uses `n`; both are defensible, the deck just doesn't say which it picked.)*

## 3.3 The covariance matrix

For `d` features, stack every pairwise covariance into a `d×d` matrix:

```
      ⎡ Var(x₁)      Cov(x₁,x₂)   ...  Cov(x₁,x_d) ⎤
Σ  =  ⎢ Cov(x₂,x₁)   Var(x₂)      ...  Cov(x₂,x_d) ⎥
      ⎢    ...          ...       ...      ...     ⎥
      ⎣ Cov(x_d,x₁)  Cov(x_d,x₂)  ...  Var(x_d)    ⎦
```

**Properties:**
- **Diagonal** = variances of each feature
- **Off-diagonal** = covariances between pairs
- **Symmetric** (`Cov(X,Y) = Cov(Y,X)`) → real eigenvalues, orthogonal eigenvectors
- **Positive semi-definite** → all eigenvalues ≥ 0
- **Scale-dependent** → this is the root of the standardisation issue in Part 4

## 3.4 Worked example

*(This is the deck's example, with the arithmetic corrected — see Part 9.)*

| Person | Height (cm) | Weight (kg) | Age (yrs) | Income ($) |
|---|---|---|---|---|
| 1 | 160 | 55 | 25 | 3000 |
| 2 | 170 | 65 | 30 | 5000 |
| 3 | 180 | 75 | 35 | 7000 |

**Step 1 — means:**
```
x̄₁ = (160+170+180)/3 = 170
x̄₂ = (55+65+75)/3    = 65
x̄₃ = (25+30+35)/3    = 30
x̄₄ = (3000+5000+7000)/3 = 5000
```

**Step 2 — deviations from the mean:**

| Person | h−h̄ | w−w̄ | a−ā | i−ī |
|---|---|---|---|---|
| 1 | −10 | −10 | −5 | −2000 |
| 2 | 0 | 0 | 0 | 0 |
| 3 | +10 | +10 | +5 | +2000 |

**Step 3 — covariances (using divisor `n = 3`):**
```
Var(height)         = ((−10)² + 0² + 10²)/3            = 200/3      =    66.67
Cov(height, weight) = ((−10)(−10) + 0 + (10)(10))/3    = 200/3      =    66.67
Cov(height, age)    = ((−10)(−5) + 0 + (10)(5))/3      = 100/3      =    33.33
Cov(height, income) = ((−10)(−2000) + 0 + (10)(2000))/3 = 40000/3   = 13333.33
Var(income)         = ((−2000)² + 0 + (2000)²)/3       = 8000000/3  = 2666666.67
```

**Step 4 — assemble:**
```
        ⎡    66.67       66.67       33.33      13333.33 ⎤
  Σ  =  ⎢    66.67       66.67       33.33      13333.33 ⎥
        ⎢    33.33       33.33       16.67       6666.67 ⎥
        ⎣ 13333.33    13333.33     6666.67    2666666.67 ⎦
```

**Two things to notice — and to raise unprompted if you're shown this example:**

1. **This dataset is degenerate.** Every variable is a perfect linear function of every other (height goes 160→170→180 in lockstep with weight 55→65→75). The covariance matrix has **rank 1**. PCA here yields exactly one non-zero eigenvalue: PC1 explains 100% of the variance and PC2, PC3, PC4 explain nothing. It's a fine arithmetic drill and a misleading statistics example.
2. **The income entries dwarf everything else** — not because income matters more, but because dollars are numerically bigger than centimetres. Exactly the trap in §4.5.

```python
import numpy as np
X = np.array([[160,55,25,3000],[170,65,30,5000],[180,75,35,7000]], float)
np.cov(X, rowvar=False, bias=True)     # bias=True → divisor n
np.linalg.matrix_rank(np.cov(X, rowvar=False))   # -> 1
```

## 3.5 Covariance vs. correlation

Covariance has ugly units (cm·kg) and unbounded magnitude, so you cannot compare covariances across pairs. **Correlation is covariance standardised:**

```
Corr(X,Y) = Cov(X,Y) / (σ_X · σ_Y)      ∈ [−1, +1]
```

> **The correlation matrix *is* the covariance matrix of the standardised data.** Remember this one line — it makes the PCA scaling question in §4.5 almost trivial to answer.

---

# Part 4 — Principal Component Analysis (PCA)

> **Priority markers.** 🔴 = you will be asked this · 🟡 = likely follow-up · ⚪ = bonus depth, only if they push.
> If you're short on prep time, master §4.1, §4.4 and §4.5. Those three carry most of the marks.

---

## 4.1 🔴 The 60-second answer

Have this ready verbatim. It's the most common opening question on the topic.

> "PCA finds a new set of axes for your data, ordered by how much variation each one captures. The first axis points along the direction the data spreads most; the second is the next-best direction perpendicular to it, and so on. Since the last few axes usually capture almost nothing, you drop them and keep a smaller set of features.
>
> Mechanically: standardise the features, compute the covariance matrix, take its eigenvectors and eigenvalues, sort by eigenvalue descending, and project the data onto the top k eigenvectors. The eigenvectors are the new directions — the principal components — and each eigenvalue is the variance captured along its direction.
>
> The main things to watch are that you have to scale first, because PCA maximises variance and variance depends on units; and that it's unsupervised, so it can discard a low-variance direction that happened to be predictive."

That last paragraph is doing real work — it pre-empts the two most common follow-ups and shows you know the failure modes without being asked.

**And for a non-technical stakeholder:**

> "You have 50 measurements per customer, and many say the same thing in different ways — annual income, monthly income and spending limit all move together. PCA builds a smaller set of summary scores that keep most of what actually varies between customers. Fifty columns become six, you lose very little, and the model trains faster and is less confused by redundancy."

---

## 4.2 🔴 Why it works — the intuition

**The picture:** your data is an elongated cloud of points. PCA finds the long axis of that cloud and makes it the new first axis, then the next-longest perpendicular direction, and so on. If the cloud is a thin cigar, one axis describes nearly everything and the rest can go.

**A concrete motivation** — say this when asked *why* you'd want it:

> "Modelling baseball players, you have both `hits` and `singles`. Every single is a hit, so they correlate around 0.98 — plot them and you get almost a straight line. They carry nearly the same information, and feeding both to a linear model creates multicollinearity. Rather than arbitrarily dropping one, PCA builds a new axis pointing *along* that line, so one number captures nearly everything both columns were saying."

Each component is a **linear combination** of the original features:
```
PC1 = 0.84 × hits + 0.54 × singles     ← the shared "volume of hitting" signal
PC2 = 0.54 × hits − 0.84 × singles     ← the residual "power hitter" signal
```

**Why bother at all — the curse of dimensionality.** More dimensions means slower training, more overfitting risk, and **distance collapse**: in high dimensions the nearest and farthest neighbours end up almost equidistant, which quietly breaks kNN, k-means and clustering. Plus multicollinearity destabilises linear-model coefficients, and you can't plot 24 dimensions.

**Two equivalent framings of the objective** — quote either; knowing they're the same scores points:
- *Maximise* the variance of the projected points, or
- *Minimise* the squared perpendicular distance from points to the new axis (reconstruction error)

⚠️ The second is **not** ordinary least squares. OLS minimises *vertical* distance (error in `y` only); PCA minimises *perpendicular* distance. OLS treats one variable as special, PCA treats them symmetrically. A favourite follow-up.

---

## 4.3 🔴 The algorithm

```
1. STANDARDISE                z = (x − mean) / std
2. COVARIANCE MATRIX of the standardised data
3. EIGENVECTORS + EIGENVALUES of that matrix
4. RANK by eigenvalue, descending
5. LOADINGS of PC1 = the top eigenvector
6. EXPLAINED VARIANCE of PC1 = λ₁ / Σλ
7. REPEAT for PC2, PC3, ...
8. PROJECT:  X_reduced = X_centred · W_k        ← W_k = top-k eigenvectors (d × k)
```

Step 8 gets left off most summaries, but it's where the reduction actually happens.

---

## 4.4 🔴 Worked example — and how to do it in 10 seconds

**Interviewers will not ask you to grind through messy decimals or use the quadratic formula by hand.** That tests nothing and burns time you need for interpretation. They hand you a deliberately friendly matrix and watch your approach. Nearly every 2×2 you'll see falls into one of three patterns.

### The example

Five students: **hours studied** (x) and **practice problems** (y).

| Student | A | B | C | D | E |
|---|---|---|---|---|---|
| **x** | 2 | 4 | 6 | 8 | 10 |
| **y** | 2 | 6 | 4 | 10 | 8 |

Both means are 6. Deviations, squares and cross-products:

| | x−x̄ | y−ȳ | (x−x̄)² | (y−ȳ)² | product |
|---|---|---|---|---|---|
| A | −4 | −4 | 16 | 16 | 16 |
| B | −2 | 0 | 4 | 0 | 0 |
| C | 0 | −2 | 0 | 4 | 0 |
| D | 2 | 4 | 4 | 16 | 8 |
| E | 4 | 2 | 16 | 4 | 8 |
| **Σ** | 0 | 0 | **40** | **40** | **32** |

Divide by `n−1 = 4`:
```
      ⎡ 10   8 ⎤
Σ  =  ⎢        ⎥          correlation = 8/√(10·10) = 0.8
      ⎣  8  10 ⎦
```

**Now spot the pattern: the diagonals are equal.** By Shortcut ① below, you write the answer down with no algebra:

```
λ = 10 + 8 = 18   and   10 − 8 = 2
eigenvectors = [1, 1]/√2  and  [1, −1]/√2
explained variance = 18/20 = 90%  and  2/20 = 10%
```

Verify in five seconds: trace `= 20 = 18 + 2` ✓, det `= 100 − 64 = 36 = 18 × 2` ✓.

**Interpretation — the part they actually score:**
```
PC1 = 0.707·hours + 0.707·problems   →  "overall effort"                    90%
PC2 = 0.707·hours − 0.707·problems   →  "style": time-heavy vs volume-heavy  10%
```
Both variances are equal and the features positively correlated, so PC1 lands exactly on the 45° line — the long axis of the cloud. Drop PC2 and you go 2 features → 1, losing 10% of the variance.

**Projecting a point.** Student A's centred vector is `[−4, −4]`:
```
PC1 score = (−4)(0.707) + (−4)(0.707) = −5.66   (well below average effort)
PC2 score = (−4)(0.707) − (−4)(0.707) =  0      (perfectly balanced style)
```

```python
X = np.array([[2,2],[4,6],[6,4],[8,10],[10,8]], float)
C = np.cov(X, rowvar=False)                    # [[10, 8], [8, 10]]
w, v = np.linalg.eigh(C); i = np.argsort(w)[::-1]
w[i], v[:, i][:, 0], (w/w.sum())[i]            # [18,2], [.707,.707], [0.9,0.1]
```

### The three shortcuts

**① Equal diagonals — the 45° trick**
```
      ⎡ v   c ⎤
Σ  =  ⎢       ⎥    →   λ = v+c  and  v−c
      ⎣ c   v ⎦        eigenvectors are ALWAYS [1,1]/√2 and [1,−1]/√2
```
The eigenvectors don't depend on the numbers at all — only the eigenvalues do. Check by inspection: `Σ[1,1]ᵀ = [v+c, c+v]ᵀ = (v+c)[1,1]ᵀ`. Geometrically, equal variances mean the cloud is symmetric about the 45° line, so the long axis has nowhere else to go.

⚠️ **The sign trap.** `[1,1]` always pairs with `v+c` — but **which is PC1 depends on the sign of `c`**:

| | λ for `[1,1]` | λ for `[1,−1]` | PC1 is |
|---|---|---|---|
| `c > 0` | `v+c` ← larger | `v−c` | **`[1,1]`** — the "overall size" direction |
| `c < 0` | `v+c` | `v−c` ← larger | **`[1,−1]`** — the "trade-off" direction |

For `Σ = [[10,−8],[−8,10]]` the eigenvalues are still 18 and 2, but `Σ[1,−1]ᵀ = 18·[1,−1]ᵀ`, so PC1 is `[1,−1]`. Sensible: if two features move oppositely, the axis of greatest spread measures their *difference*.

**② Zero off-diagonal — the diagonal give-away**

For `Σ = [[9,0],[0,4]]`, the eigenvalues are just the diagonal entries: `λ = 9, 4`, with eigenvectors `[1,0]` and `[0,1]`. (True of any triangular matrix.)

⚠️ **This is a trap with a conceptual answer attached.** They want you to notice what it *means*:

> "Zero covariance means the features are already uncorrelated, so the cloud is already aligned with the axes. PCA has nothing to rotate — the components *are* the original features, just relabelled in variance order. Running PCA here buys nothing, which makes me ask whether we need reduction at all or should simply drop the low-variance features."

Watch the ordering: for `[[3,0],[0,7]]`, PC1 is `[0,1]` — the **second** feature, since 7 > 3.

**③ Trace and determinant — guess the factors**

The general method. From §2.5: `λ₁+λ₂ = trace` and `λ₁λ₂ = det`. So instead of expanding the characteristic polynomial, ask: **what two numbers add to the trace and multiply to the determinant?** Interview matrices are built to have integer answers.

```
      ⎡ 7   2 ⎤     trace = 7 + 4 = 11
Σ  =  ⎢       ⎥     det   = 28 − 4 = 24        →  8 and 3
      ⎣ 2   4 ⎦     λ = 8, 3      explained: 73% / 27%
```

| Σ | trace | det | factors | λ | PC1 var |
|---|---|---|---|---|---|
| `[[7,2],[2,4]]` | 11 | 24 | 8 × 3 | **8, 3** | 73% |
| `[[5,2],[2,5]]` | 10 | 21 | 7 × 3 | **7, 3** | 70% |
| `[[4,1],[1,4]]` | 8 | 15 | 5 × 3 | **5, 3** | 63% |
| `[[6,2],[2,3]]` | 9 | 14 | 7 × 2 | **7, 2** | 78% |
| `[[10,8],[8,10]]` | 20 | 36 | 18 × 2 | **18, 2** | 90% |

**Fallback:** `λ = ½[T ± √(T² − 4D)]`. For `[[6,2],[2,3]]`: `√(81−56) = 5` → `(9±5)/2 = 7, 2` ✓. If the discriminant isn't a perfect square, say so out loud — it usually means an arithmetic slip, and flagging it beats silently producing decimals.

**Always sanity-check** with `Σλ = trace` and `Πλ = det`. Five seconds, catches sign errors, and doing it visibly signals rigour.

**Then the eigenvectors — fast.** Plug `λ` into `(Σ − λI)v = 0` and read **only the first row**; the second is guaranteed redundant (that's what `det = 0` means).
```
Σ = [[7,2],[2,4]], λ₁ = 8:   (Σ − 8I) = ⎡ −1   2 ⎤   row 1: −x + 2y = 0 → v₁ = [2, 1]ᵀ
                                        ⎣  2  −4 ⎦
```
Leave it as `[2,1]` and say "up to scaling" — that's complete. Normalise to `[2,1]/√5` only if asked.

> **Spend the time you saved on:** the explained-variance ratio, what the components *mean*, and raising the scaling question unprompted. That's what's being scored.

---

## 4.5 🔴 The scaling trap — the highest-yield section here

**This is the follow-up you will get.**

### The problem

The covariance matrix is **scale-dependent**. A feature in dollars has variance in the millions; the same information in thousands-of-dollars has variance six orders of magnitude smaller. PCA maximises variance — so **it hands PC1 to whichever feature has the biggest units**, regardless of importance.

### The demonstration

500 people; height (cm), weight (kg), income ($). Height and weight correlate strongly; income is independent noise.

| | column variances | explained variance ratio | PC1 loadings |
|---|---|---|---|
| **Raw (covariance matrix)** | `[92, 100, 234,000,000]` | `[1.00, 0.00, 0.00]` | `[0.00, 0.00, 1.00]` ❌ |
| **Standardised (correlation matrix)** | `[1, 1, 1]` | `[0.62, 0.33, 0.05]` | `[0.71, 0.71, 0.06]` ✅ |

On raw data, PCA reports one component explaining 100% of the variance. It has learned **nothing** except that dollars are numerically large; the real height–weight structure is invisible. Standardised, PC1 correctly recovers the body-size factor.

**Worse: the answer changes with units.** Re-express height in metres instead of centimetres and the raw loadings change again. *Same data, same information, different answer.*

### The rule

| Situation | Use |
|---|---|
| Features in **different units** (age, income, counts, %) | **Standardise** → PCA on the **correlation** matrix. The default. |
| Features in the **same unit and comparable scale** (pixels 0–255; gene expression on one platform) | Covariance matrix is defensible — variance differences are real signal, not unit artefacts |
| Unsure | **Standardise** |

> **The one-line answer:** *"Yes — PCA maximises variance and variance is scale-dependent, so without standardising, the feature with the largest units takes PC1 regardless of importance. Standardising first is equivalent to running PCA on the correlation matrix instead of the covariance matrix."*

⚠️ **Fit the scaler on train only** — `scaler.fit(X_train)` then `.transform(X_test)`. Fitting on all data leaks test means and standard deviations into training. Same for `pca.fit()`. Use a `Pipeline` and it can't happen.

---

## 4.6 🔴 Choosing the number of components, k

No formula. Four defensible approaches:

1. **Cumulative explained variance** (most common) — smallest `k` reaching 90/95/99%. `PCA(n_components=0.95)` does it for you.
2. **Scree plot / elbow** — plot eigenvalues descending, find where the curve flattens. Subjective but fast and often unambiguous.
3. **Kaiser criterion** — on the **correlation** matrix, keep `λ > 1`: each standardised variable contributes variance 1, so a component explaining less than one variable's worth isn't earning its place. Only valid on the correlation matrix; widely criticised as over-retaining.
4. **Downstream performance** — treat `k` as a hyperparameter and cross-validate the actual model.

> **The answer that scores:** *"Cumulative variance as a sanity check, but ultimately I'd tune k by cross-validating the downstream model. 95% of variance isn't inherently meaningful — variance isn't the same as predictive signal, and a component can carry very little variance and still be the one that separates the classes."*

Worked: given 40/25/15/10/5/5%, to reach ≥80% you need 40+25+15 → **k = 3**.

---

## 4.7 🟡 What the components actually mean

| Term | What it is | Shape | sklearn |
|---|---|---|---|
| **Loadings** (components) | The eigenvectors — the *recipe* mapping features → PC | `k × d` | `pca.components_` |
| **Scores** | The transformed data — each row's *coordinates* | `n × k` | `pca.transform(X)` |
| **Explained variance** | The eigenvalues | `k` | `pca.explained_variance_` |
| **Explained variance ratio** | λᵢ / Σλ | `k` | `pca.explained_variance_ratio_` |

Loadings answer *"what does PC1 mean?"*; scores answer *"where does customer #457 sit?"* A **biplot** shows both — points are scores, arrows are loadings.

**But say this before they ask:**

> **Principal components are linear combinations of original features, and usually have no business meaning.**

`PC1 = 0.31·age − 0.22·income + 0.44·tenure + ...` isn't something you put in front of a stakeholder or a regulator. Sometimes a component is obviously interpretable — a "body size" factor — but that's a happy accident.

**In regulated settings** (credit, insurance, healthcare) this is a blocker: if you need adverse-action reasons, you can't tell a declined applicant "your PC3 was too low." A real reason to prefer feature *selection* over feature *extraction*. If you need both, **Sparse PCA** forces most loadings to zero so each component involves only a few original features.

*(Sign note: eigenvector signs are arbitrary, so loadings can flip between runs. Only relative signs within a component are meaningful.)*

---

## 4.8 🔴 When PCA breaks — and what to use instead

| Assumption / situation | Problem | Alternative |
|---|---|---|
| **Relationships are linear** | Only finds linear structure; fails on curved manifolds | Kernel PCA, t-SNE, UMAP, autoencoder |
| **Variance = information** | **Unsupervised — never sees `y`.** Can discard the low-variance direction that separates your classes | **LDA** (supervised), **PLS** |
| **No extreme outliers** | Squared deviations mean one outlier can rotate PC1 toward it | Robust PCA; treat outliers first |
| **Features are correlated** | If already uncorrelated, PCA just reorders your columns | Check the correlation matrix first — maybe you don't need it |
| **Orthogonality is appropriate** | If true latent factors are correlated, forcing orthogonality distorts them | Factor analysis (oblique rotation), ICA |
| **Categorical / one-hot data** | Variance isn't meaningful for indicators | MCA, FAMD, embeddings |
| **Sparse high-dim (TF-IDF)** | Centring destroys sparsity, can exhaust memory | **TruncatedSVD** (§5.4) |
| **Interpretability required** | Components have no business meaning | Feature selection, **Sparse PCA** |
| **Tree-based downstream model** | Trees handle correlated/irrelevant features fine; PCA destroys axis-aligned splits | Often just skip PCA |
| **Data too large for memory** | — | **Incremental PCA** (mini-batches), **Randomised PCA** (`svd_solver='randomized'`) |

**Applications where it does work well:** noise reduction (noise lives in low-variance components), visualisation, image compression, eigenfaces (a 128×128 face is 16,384 dims; ~20 eigenvectors capture most of what distinguishes faces), anomaly detection via reconstruction error, and speeding up training.

*(`PCA(whiten=True)` additionally rescales each component to unit variance — useful when a downstream model assumes isotropic inputs, harmful when relative component scale is real signal.)*

---

## 4.9 ⚪ If they push deeper

**Why eigenvectors? The derivation.** Very few candidates can do this; it takes 90 seconds.

Maximise the variance of data projected onto a unit vector `w`. That variance is `wᵀΣw`, subject to `wᵀw = 1`. Lagrangian:
```
L(w, λ) = wᵀΣw − λ(wᵀw − 1)

∂L/∂w = 2Σw − 2λw = 0     ⟹     Σw = λw        ← the eigenvector equation
```
Substituting back: `Var(Xw) = wᵀΣw = wᵀ(λw) = λ`.

> **The variance along an eigenvector is exactly its eigenvalue.** So maximising variance means taking the largest one. This single result explains why eigenvectors, why sorted by eigenvalue, and why explained variance is `λᵢ/Σλ`.

**Does PCA remove multicollinearity?** Completely — components are uncorrelated by construction (`WᵀΣW = Λ` is diagonal). Regression on PC scores is **Principal Component Regression (PCR)**. The catch: PCA picks components by variance without looking at `y`, so PCR can discard a low-variance but highly predictive direction. **Partial Least Squares (PLS)** fixes this by choosing directions that maximise covariance *with the target*.

**Complexity.** `O(nd²)` to form the covariance matrix, `O(d³)` to eigendecompose. That `d³` is why the covariance route is impractical for large `d` — and one reason sklearn uses SVD (§5.3). Randomised SVD for top-k only: `≈ O(ndk)`.

**Missing values.** No native handling; sklearn errors. Impute first (median, or iterative/MICE), or use probabilistic PCA with EM, which handles missingness properly. **Never fill with zeros on unstandardised data** — zero isn't neutral, it's an extreme value that drags components toward it.

**Test set?** `transform` with the *fitted* PCA, never `fit_transform`.

---

## 4.10 🟡 Code

```python
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Pipeline: scaler and PCA fit on training folds only. No leakage.
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("pca",   PCA(n_components=0.95, random_state=0)),   # keep 95% of variance
    ("clf",   LogisticRegression(max_iter=1000)),
]).fit(X_train, y_train)

pca = pipe.named_steps["pca"]
pca.n_components_                                   # how many were kept
np.cumsum(pca.explained_variance_ratio_)            # cumulative curve

# Which original features drive PC1?
pd.Series(pca.components_[0], index=feature_names).abs().nlargest(10)

# How much did we actually lose?
Xs = StandardScaler().fit_transform(X_train)
p  = PCA(n_components=6).fit(Xs)
((Xs - p.inverse_transform(p.transform(Xs))) ** 2).mean()
```

**From scratch — know how to write this:**
```python
def pca_from_scratch(X, k):
    Xc     = X - X.mean(axis=0)              # 1. centre
    cov    = np.cov(Xc, rowvar=False)        # 2. covariance (n−1 divisor)
    w, v   = np.linalg.eigh(cov)             # 3. eigendecompose (symmetric → eigh)
    idx    = np.argsort(w)[::-1]             # 4. rank descending (eigh gives ascending)
    w, v   = w[idx], v[:, idx]
    W_k    = v[:, :k]                        # 5. top-k eigenvectors
    return Xc @ W_k, W_k, w / w.sum()        # 6. project
```

> ⚠️ Use `np.linalg.eigh`, not `eig`, for covariance matrices — it exploits symmetry, is faster, and returns **real** eigenvalues in **ascending** order. `eig` can return tiny imaginary components from floating-point noise.


---

# Part 5 — Singular Value Decomposition (SVD)

## 5.1 The definition

SVD factorises **any** matrix — rectangular, non-square, non-symmetric — into three matrices:

```
M  =  U  Σ  Vᵀ
```

| Matrix | Shape (full SVD) | What it is |
|---|---|---|
| **M** | `m × n` | Original data (e.g. rows = users, cols = movies) |
| **U** | `m × m` | **Left singular vectors** — orthogonal; columns are eigenvectors of `MMᵀ` |
| **Σ** | `m × n` | Diagonal matrix of **singular values**, sorted **descending** |
| **Vᵀ** | `n × n` | **Right singular vectors** — orthogonal; rows are eigenvectors of `MᵀM` |

**Singular values are the square roots of the eigenvalues of `MᵀM`** (equivalently `MMᵀ`), and they are always **non-negative** and sorted **largest first**.

⚠️ Note the shapes: in the **full** SVD, `U` is `m × m` and `Vᵀ` is `n × n`. In the **reduced (thin)** SVD — what `numpy.linalg.svd(M, full_matrices=False)` returns — `U` is `m × r` and `Σ` is `r × r` where `r = min(m,n)`. *(The source deck states `U` is `m × n` in text but draws `m × m`; the diagram is the full SVD and is the correct one.)*

## 5.2 Intuition

Think of SVD as a **simplification process** — decomposing a complex system into simpler, ranked parts. The Lego analogy from the deck works well: a messy pile of bricks becomes neat stacks, sorted by colour and size.

```
M   = the original data, in full complexity
U   = the major patterns  ("what are the themes?")
Σ   = the importance of each pattern  ("how much does each theme matter?")
Vᵀ  = how the patterns relate to the original columns  ("what is each theme made of?")
```

Geometrically, **any linear transformation is a rotation, then a scaling, then another rotation** — that's exactly `V ᵀ` (rotate), `Σ` (scale each axis), `U` (rotate again). Every matrix does this, which is why SVD always exists.

## 5.3 The PCA ↔ SVD relationship — expect this question

**They compute the same thing.** If `X` is your mean-centred data matrix (`n` rows × `d` features):

```
Covariance:  Σ_cov = XᵀX / (n − 1)

SVD of X:    X = U S Vᵀ
So:          XᵀX = (U S Vᵀ)ᵀ (U S Vᵀ) = V S Uᵀ U S Vᵀ = V S² Vᵀ     (since UᵀU = I)

Therefore:   Σ_cov = V · [S²/(n−1)] · Vᵀ
```

Compare with the eigendecomposition `Σ_cov = W Λ Wᵀ` and read off:

| PCA quantity | SVD equivalent |
|---|---|
| Principal components (eigenvectors `W`) | **`V`** (right singular vectors) |
| Eigenvalues `λᵢ` | **`sᵢ² / (n − 1)`** |
| Singular value `sᵢ` | **`√(λᵢ · (n−1))`** |
| PC scores (projected data) | **`U S`** |

*(The personal notes phrase this as "singular values are sqrt of eigenvalues multiplied by n−1" — correct, reading it as `√(λ·(n−1))`.)*

Verified numerically:
```python
Xc = X - X.mean(0);  n = len(Xc)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
eig = np.sort(np.linalg.eigvalsh(np.cov(Xc, rowvar=False)))[::-1]
np.allclose(eig, S**2 / (n-1))          # True
np.allclose(S,   np.sqrt(eig*(n-1)))    # True
```

### So why does sklearn use SVD internally?

**Three reasons, and knowing them is a strong signal of depth:**

1. **Numerical stability.** Forming `XᵀX` **squares the condition number** of the problem. If `X` is ill-conditioned, small floating-point errors get amplified badly. SVD works on `X` directly and never forms the covariance matrix.
2. **Efficiency when `d ≫ n`.** With 10,000 features and 100 samples, the covariance matrix is 10,000×10,000 — expensive and rank-deficient. SVD sidesteps it.
3. **It works on non-square, non-symmetric matrices.** Eigendecomposition needs a square matrix; SVD doesn't. That's what lets it operate on raw user–item or document–term matrices.

> **The soundbite:** *"PCA is eigendecomposition of the covariance matrix; SVD applied to the centred data matrix gives you the same components without ever forming that covariance matrix — which is more numerically stable and cheaper when features outnumber samples. sklearn's PCA is implemented via SVD for exactly that reason."*

**The one distinction that matters practically:** PCA **centres** the data (subtracts the mean). Plain SVD does not. `TruncatedSVD` in sklearn does *not* centre — which is a feature, not a bug (§5.4).

## 5.4 Truncated SVD

Keep only the top `k` singular values and their vectors:
```
M  ≈  U_k Σ_k V_kᵀ        where k ≪ min(m, n)
```

This is the **best possible rank-`k` approximation of `M`** in a least-squares sense (the Eckart–Young theorem) — a genuinely strong guarantee, and a nice thing to be able to name.

**How it works, step by step:**
1. Construct the data matrix
2. Decompose using SVD
3. Select the top `k` components (they're already sorted descending)
4. Create a reduced representation
5. Transform the data
6. Use downstream in ML

⚠️ **The singular values in `Σ` are sorted in *decreasing* order, not increasing.** *(The personal notes have this backwards — worth correcting in your copy.)*

**When to use TruncatedSVD instead of PCA:**

| | PCA | TruncatedSVD |
|---|---|---|
| Centres the data | Yes | **No** |
| Works on sparse matrices | No (centring destroys sparsity) | **Yes** |
| Typical use | Dense numeric features | **TF-IDF / count matrices, recommender data** |

> On a TF-IDF matrix with 50,000 vocabulary terms, centring turns a sparse matrix into a dense one and may exhaust memory. `TruncatedSVD` keeps it sparse. **Applying TruncatedSVD to TF-IDF is called Latent Semantic Analysis (LSA).**

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

X_tfidf = TfidfVectorizer(max_features=20000).fit_transform(docs)   # sparse
lsa     = TruncatedSVD(n_components=100, random_state=0)
X_lsa   = lsa.fit_transform(X_tfidf)          # dense, 100-dim "topics"
print(lsa.explained_variance_ratio_.sum())
```

## 5.5 SVD applications

- **Image compression** — reconstruct with `k` singular values; at `k=2` the image is unrecognisable, by `k≈50` it's close to the original, and by `k≈250` visually identical. Storage drops from `m·n` to `k(m+n+1)` numbers.
- **LSA / topic modelling** — latent "topics" in a document–term matrix
- **Recommender systems** — latent factors in a user–item matrix (Part 6)
- **Word embeddings** — SVD on a word co-occurrence matrix gives dense word vectors; this is the ancestor of word2vec and the mechanism behind GloVe
- **Noise reduction & denoising** — small singular values usually encode noise
- **Pseudo-inverse & least squares** — the numerically stable way to solve ill-conditioned linear systems
- **Matrix rank estimation** — count singular values above a tolerance

## 5.6 Word vectors via SVD — the LSA pipeline

From the personal notes, worth reconstructing because it links SVD to NLP:

1. **Build a co-occurrence matrix `X`** (`n × n`, `n` = vocabulary size). `X[i][j]` counts how often word `wᵢ` appears within a context window of word `wⱼ` — e.g. within 5 words either side.
2. **Problem:** `n` is the vocabulary size, so `X` is enormous. **Fix:** keep only the top `m` most important words, selected via TF-IDF.
3. **Decompose:** `X = U Σ Vᵀ`
4. **Truncate:** keep the top `k` singular values, discard the rest.
5. **Result:** each word now has a dense `k`-dimensional vector instead of a sparse `n`-dimensional row.

This family — **LSA, pLSA, LDA, GloVe** — all learn word vectors by factorising a co-occurrence statistic. The main conceptual step from here to word2vec is replacing the explicit matrix factorisation with a predictive objective trained by SGD (and it was later shown that word2vec is implicitly factorising a shifted PMI matrix — the ideas are closer than they look).

---

# Part 6 — Matrix Factorization & Recommender Systems

# Recommendation Systems Cheat Sheet

## 👥 1. Collaborative Filtering (Behavior-Based)
*Core Concept: Analyzes patterns in user behaviors, clicks, and ratings without needing to know what the items actually are.*

### A. User-User Collaborative Filtering
* **How it works:** Finds users with similar rating histories to the target user and recommends items those lookalike peers enjoyed.
* **Pros/Cons:** Highly personalized but scales poorly as user numbers grow.
#### 👥 Handling Cold-Starts in User-User CF

##### 1. When a New User is Added (User Cold-Start)
* Pure User-User CF fails because a blank history cannot be mathematically compared to other users.
* **Fix 1 (Demographic Matching):** Use onboarding data (age, location, selected preferences) to temporarily group the user with an existing cluster of similar peers.
* **Fix 2 (Popularity Default):** Serve high-engagement, trending items to harvest the user's first few organic ratings.

##### 2. When a New Item is Added (Item Cold-Start)
* Pure User-User CF completely misses new items because zero users have it in their history to pass along to lookalike peers.
* **Fix 1 (Content-Based Bridge):** Match the item's metadata directly to users with a strong historical affinity for that specific genre/tag.
* **Fix 2 (Power-User Seeding):** Intentionally inject the unrated item into the feeds of highly active users to fast-track the collection of initial ratings.

### B. Item-Item Collaborative Filtering
* **How it works:** Measures similarity between items based on how frequently the same people rate or buy them together.
* **Pros/Cons:** Computationally stable and efficient since item catalogs change slower than user bases.
#### 🆕 Handling New Users in Item-Item CF (User Cold-Start)
* Pure Item-Item CF fails on new users because there is no interaction history to match against the item similarity matrix.
* **Fix 1 (Onboarding):** Force user to pick 3 favorite items/genres at signup to seed their history.
* **Fix 2 (Fallback):** Serve globally popular/trending items until the first organic click or rating occurs.
#### 📦 Handling New Items in Item-Item CF (Item Cold-Start)
* Pure Item-Item CF cannot recommend new items because they have zero user ratings/history to compute similarity.
* **Fix 1 (Content-Based):** Seed the initial similarity scores using metadata (genres, description tags) instead of user behavior.
* **Fix 2 (Exploration Boosting):** Intentionally inject new items into random user feeds to gather the first crucial interaction data.
* **Fix 3 (Parent Inheritance):** Force the item to temporarily inherit the similarity scores of its overarching category or brand.

### C. Matrix Factorization (Latent-Factor Model)
* **How it works:** Compresses the raw user-item matrix into hidden, mathematical dimensions to predict missing ratings via quick dot products.
* **Pros/Cons:** Handles sparse data brilliantly and calculates recommendations in microseconds, but lacks explainability.
#### 📉 Handling Cold-Starts in Matrix Factorization (MF)

##### 1. When a New User is Added (User Cold-Start)
* Pure MF fails because the new user lacks a latent vector ($b_i$) in User Matrix $B$ to compute dot products.
* **Fix 1 (Average Vector):** Initialize the user with a global average vector or a vector representing their demographic cluster.
* **Fix 2 (Folding-In / Projection):** Use onboarding selections to mathematically project and calculate a temporary vector ($b_i$) on the fly without running a full model retrain.

##### 2. When a New Item is Added (Item Cold-Start)
* Pure MF fails because the new item lacks a latent vector ($c_j$) in Item Matrix $C$.
* **Fix 1 (Metadata Mapping):** Train a secondary model to convert item descriptions and genres into an estimated latent vector ($c_j$) based on similar items.
* **Fix 2 (Average + Explore):** Give the item an average item vector and intentionally inject it into diverse user feeds to harvest the first training ratings.

---

## 🏷 2. Content-Based Filtering (Attribute-Based)
*Core Concept: Completely ignores other users and focuses entirely on matching item traits to a single user's profile.*

* **How it works:** Recommends items that share text keywords, genres, actors, or descriptions with items the target user has explicitly liked before.
* **Pros/Cons:** Solves the cold-start problem for new items, but limits discovery by trapping users in an echo chamber of things they already know they like.

# 📈 Recommendation System Evaluation Metrics

## 🧮 1. Rating Accuracy Metrics
*Focus: Measuring how close predicted rating numbers are to the actual user ratings.*

* **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual ratings. Simple and interpretable.
* **RMSE (Root Mean Squared Error):** Squares errors before averaging, heavily penalizing large misses. The standard metric used for traditional Matrix Factorization algorithms.

## 🏅 2. Top-K Ranking Metrics
*Focus: Measuring if the model puts the absolute best items at the very top of the user's feed.*

* **Precision@K:** The percentage of recommended items in the top $K$ list that the user actually liked (e.g., 3 out of 5 hits = 60% Precision@5).
* **Recall@K:** The percentage of total relevant items in the catalog that the model managed to capture inside the top $K$ recommendations.
* **MAP (Mean Average Precision):** Computes precision across multiple recommendation lengths, penalizing the system if relevant items drop lower down the list.
* **NDCG (Normalized Discounted Cumulative Gain):** The industry standard. Uses a logarithmic discount to give maximum credit for relevant items placed at rank #1, and significantly less credit for items placed lower down.

## 6.1 What matrix factorization is

> Decomposing a matrix into a product of two or more lower-rank matrices — `A ≈ B · Cᵀ` — to uncover **latent features** in the data.

If `A` can be written as the product `BCD`, that's a **multiplicative decomposition**, and `B`, `C`, `D` are **factors** of `A`. PCA (eigendecomposition of the covariance matrix) and SVD are both instances of matrix factorization.

**The family:**

| Method | Key property | Typical use |
|---|---|---|
| **SVD** | Orthogonal factors, unique, best rank-k | General DR, LSA |
| **Truncated SVD** | Top-k only | Sparse text, recommenders |
| **NMF** (Non-negative MF) | All entries ≥ 0 | Topic modelling, image parts, spectra |
| **PMF** (Probabilistic MF) | Probabilistic framing | Recommenders |
| **ALS** (Alternating Least Squares) | Optimisation method | Large-scale recommenders (Spark) |
| **Low-Rank Matrix Completion** | Fills missing entries | Recommenders |
| **Tensor Factorization** | Generalises to >2 dimensions | Context-aware recommenders |

**Why NMF is special:** because everything stays non-negative, the factors are **additive parts** — you can't cancel one component against another. On faces this yields recognisable noses and eyes rather than ghostly full-face templates; on documents it yields topics as positive word mixtures. **NMF is often more interpretable than SVD**, and that's the reason to reach for it.

## 6.2 The recommender setup

Rows = users, columns = items, cells = ratings. Real matrices are >99% empty (`sparsity = empty cells / total cells`).

|  | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| **U1** | | 5 | 4 | 2 | 1 |
| **U2** | 1 | | | 5 | 3 |
| **U3** | 1 | 4 | 4 | 1 | |
| **U4** | | | 2 | | 2 |
| **U5** | 3 | 1 | 1 | | |

**The task:** U4 hasn't rated M4. Would they like it?

**Three approaches:**

- **Content-based** — ignores rating patterns; uses item *features* (genre, director, price, description) to recommend items similar to ones the user liked. ✅ No cold start for new items. ❌ Needs good metadata; over-specialises.
- **Collaborative filtering (CF)** — *"users who agreed in the past tend to agree in future."* Uses only the rating matrix. ✅ No metadata needed, finds surprising recommendations. ❌ Cold start; sparsity.
- **Hybrid** — what production systems actually do.

## 6.3 User–user vs item–item similarity

**User–user:** build the user–item matrix → compute a similarity matrix between users (cosine/Pearson) → find users most similar to the target → recommend items they liked that the target hasn't seen.
❌ **User taste drifts.** You buy a watch one day and camping gear the next, so user vectors are unstable.

**Item–item:** find items similar to those the target already liked, and recommend those.
✅ **Item rating profiles are far more stable over time** — after an initial settling period a film's ratings barely move.

> **Rule of thumb worth quoting:** *"If users vastly outnumber items and item ratings are stable over time, item–item beats user–user."* With 50M users and 10K products the item similarity matrix is 10K×10K rather than 50M×50M, and it only needs recomputing nightly. This is why Amazon's published algorithm is item–item.

## 6.4 Matrix factorization for CF — the latent-factor model

Similarity methods work on the raw matrix. **Matrix factorization instead learns latent features.**

Decompose the ratings matrix `A` into `B · Cᵀ`:

```
A (n users × m items)  ≈  B (n × d)  ·  Cᵀ (d × m)
```

- **`B`** has `n` rows (users) and `d` columns → a **`d`-dimensional feature vector `bᵢ` for each user**
- **`C`** has `m` rows (items) and `d` columns → a **`d`-dimensional feature vector `cⱼ` for each item**
- `d` is the number of **latent factors** — a hyperparameter

A predicted rating is then just a dot product:
```
Âᵢⱼ = bᵢᵀ cⱼ
```

**What the latent factors mean:** nothing explicitly — they're learned, not specified. But they often *emerge* as recognisable concepts: a "gritty vs. feel-good" axis, an "action vs. dialogue" axis. A user's vector says how much they want each; an item's vector says how much it has of each. In the table above, U1 and U3 both rate Movie2 and Movie3 highly — MF discovers this shared preference direction and uses it to predict.

**Sharing Knowledge: The algorithm does not look at your 5 ratings (out of 100 movies available) in isolation. If you rated Movie X highly, it finds thousands of other users who also liked Movie X. It then looks at what those users rated highly among the remaining 95 movies to build your recommendations


## 6.5 The optimisation

> **Find `B` and `C` minimising squared error over the entries of `A` that are actually observed:**

```
min over B, C:    Σ         (Aᵢⱼ − bᵢᵀcⱼ)²   +   λ(‖B‖² + ‖C‖²)
                (i,j) ∈ observed                  └── regularisation ──┘
```

This is just a regression problem — minimising squared loss, exactly like `(yᵢ − ŷᵢ)²` with `ŷ = wᵀx`.

⚠️ **Critical detail: the sum runs only over observed entries.** The source deck says missing values "would be filled with 0" — **this is the wrong approach for a ratings recommender, and interviewers do probe it.** Filling with 0 tells the model "this user actively rated this item zero", when the truth is "we don't know". You end up training the model to reproduce your imputation rather than to predict preference. Modern MF (Funk SVD / the Netflix Prize approach) sums only over known ratings.

**The nuance:** for *implicit* feedback (clicks, views, purchases), treating missing as zero *is* often correct — a non-click is weak negative evidence. That's the basis of the Hu-Koren-Volinsky weighted-ALS method. Knowing when zero-filling is right and when it's wrong is a genuinely good answer.

**How it's solved:**
- **SGD** — iterate over observed ratings, update `bᵢ` and `cⱼ` by gradient. Simple, fast, the Netflix Prize workhorse.
- **ALS** — fix `C`, solve for `B` in closed form; fix `B`, solve for `C`; repeat. Each half is a least-squares problem. **Parallelises well** → the standard choice at scale (Spark MLlib).

**Then do matrix completion.** Once you have `B` and `C`, compute `Â = BCᵀ` — now every cell is filled with a predicted rating, including all the originally empty ones. Rank each user's unseen items by predicted rating and recommend the top ones.

## 6.6 Choosing d (the number of latent factors)

Same logic as choosing `k` in PCA: plot reconstruction error against `d` and look for the **inflection point**. Typically error falls steeply up to around `d = 10`, then flattens. Validate on held-out ratings (RMSE) rather than trusting the elbow alone.

## 6.7 The cold-start problem

**New user or new item → no historical interactions → all zeros → factorization has nothing to work with.**

**Mitigations:**
- **New user:** recommend globally popular items; or use metadata available at signup — **geolocation, browser, device**. ("People in CA on an iPad using Safari tend to buy X.")
- **New item:** use item metadata — category, price, description — and find users who bought similar items. This is content-based filtering stepping in exactly where CF fails.
- **Hybrid systems** exist largely to cover this gap.
- **Ask explicitly** — the onboarding "pick 3 genres you like" flow is a cold-start fix.

## 6.8 Clustering as matrix factorization

A neat conceptual link that shows up in stronger interviews: **k-means can be written as a matrix factorization.**

Define an assignment matrix `Z` of shape `k × n` (`k` clusters, `n` points):
- `Zᵢⱼ = 1` if point `j` belongs to centroid `i`, else 0
- Each **column** has exactly one 1 (every point is in exactly one cluster)
- Total number of 1s = `n`

The k-means objective then becomes minimising `‖X − MZ‖²` where `M` holds the centroids — a matrix factorization with a hard constraint that `Z` is binary and column-stochastic. **NMF is the relaxed version** of this, allowing soft, non-negative assignments — which is exactly why NMF often behaves like soft clustering.

## 6.9 Limitations of matrix factorization

| Limitation | Detail |
|---|---|
| **Linearity** | Assumes a linear (dot-product) relationship between latent factors. Neural CF and two-tower models exist to relax this |
| **Cold start** | New users/items have no data (§6.7) |
| **Sparsity** | Extremely sparse matrices are hard to factorise reliably |
| **Interpretability** | Latent factors usually have no clean meaning |
| **Overfitting** | Especially when `d` is large relative to the number of observed ratings → regularise |
| **Loss of context** | Time, location, device, session don't fit the plain 2-D user×item framing → needs tensor factorization or feature-based models |
| **Popularity bias** | Tends to over-recommend already-popular items, reinforcing a feedback loop |

---
**Real-World Solution: Hybrid Systems** 
```text
                  ┌──────────────────────┐
                  │   Raw User & Item    │
                  │     Interactions     │
                  └──────────┬───────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │   Matrix    │  │  Content-   │  │ Item-to-Item│
     │Factorization│  │    Based    │  │ Similarity  │
     └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
            │                │                │
            └────────────────┼────────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Ensemble / Ranker    │───► Final Top 10
                  │ (Deep Learning Model)│     Recommendations
                  └──────────────────────┘

```
# Part 7 — Non-linear reduction: t-SNE & UMAP

> 🔴 = expect it · 🟡 = likely follow-up · ⚪ = bonus depth
> If you learn one thing here: **these are visualisation tools, not preprocessing steps**, and the things people naturally read off the plots — cluster sizes and the gaps between clusters — are exactly the things you must not trust.

---

## 7.1 🔴 Why linear methods aren't enough

PCA can only rotate and project. If your data lies on a **curved manifold**, no rotation untangles it.

The standard picture: points arranged in two concentric rings, or a spiral, or a Swiss roll. The structure is obvious to the eye, but there is no straight line you can project onto that separates the rings — any linear projection overlaps them. PCA will faithfully report the directions of greatest variance and completely miss the pattern.

**The manifold assumption:** high-dimensional data often lies on (or near) a much lower-dimensional curved surface embedded in that space. A 784-pixel MNIST digit lives in 784 dimensions, but the set of *plausible handwritten digits* is a far smaller curved region — stroke thickness, slant, curvature are only a handful of underlying degrees of freedom. **Manifold learning** methods try to "unroll" that surface.

| | PCA | t-SNE / UMAP |
|---|---|---|
| Finds | Global linear directions of variance | Local neighbourhood structure |
| Preserves | Large pairwise distances | Who is near whom |
| Output for new data | `transform()` — a fixed matrix | t-SNE: none · UMAP: yes |
| Deterministic | Yes (up to sign) | **No** |
| Primary use | Preprocessing + visualisation | **Visualisation** |

---

## 7.2 🔴 t-SNE — how it works

**t-distributed Stochastic Neighbor Embedding** (van der Maaten & Hinton, 2008). Four steps:

**1. Turn high-dimensional distances into probabilities.** For each point `i`, place a Gaussian centred on it and ask: *if `i` picked a neighbour at random in proportion to that Gaussian, how likely is it to pick `j`?* That gives a conditional probability `p_{j|i}`.

The Gaussian's bandwidth `σᵢ` is set **per point** so that the entropy of its neighbour distribution matches a user-chosen **perplexity**. This is the clever bit: in a dense region `σᵢ` shrinks, in a sparse region it grows, so "neighbourhood" adapts to local density automatically.

**2. Symmetrise.** `p_{ij} = (p_{j|i} + p_{i|j}) / 2n`, giving a single joint probability per pair.

**3. Define low-dimensional similarities with a Student-t distribution** (1 degree of freedom, i.e. Cauchy):
```
q_{ij}  ∝  (1 + ‖yᵢ − yⱼ‖²)⁻¹
```

**4. Minimise `KL(P ‖ Q)`** over the low-dimensional positions `y` by gradient descent.

# t-SNE Step-by-Step Numerical Example
Calculate distance in high dim-> calculate Conditional Probabilities using exp ->Symmetrize Probabilities (p_ij)->Low-Dimensional Initialization -> Low-Dimensional Probabilities with student dist->Gradient Descent Optimization

## Setup Data
Three 1D data points:
* x1 = 1.0
* x2 = 2.0 (close to x1)
* x3 = 5.0 (far from x1 and x2)

---

## Step 1: High-Dimensional Squared Distances
Formula: d^2_ij = ||x_i - x_j||^2

* d^2_12 = (1 - 2)^2 = 1.0
* d^2_13 = (1 - 5)^2 = 16.0
* d^2_23 = (2 - 5)^2 = 9.0

---

## Step 2: High-Dimensional Conditional Probabilities (p_j|i)
Formula: p_j|i = exp(-d^2_ij / 2σ^2) / Sum_k[exp(-d^2_ik / 2σ^2)]
Assumed fixed variance: 2σ^2 = 1.0 (Numerator becomes exp(-d^2_ij))

* From x1's perspective:
  - exp(-1.0) = 0.368
  - exp(-16.0) = 0.000
  - Sum = 0.368
  - p_2|1 = 0.368 / 0.368 = 1.000
  - p_3|1 = 0.000 / 0.368 = 0.000

* From x2's perspective:
  - exp(-1.0) = 0.368
  - exp(-9.0) = 0.0001
  - Sum = 0.3681
  - p_1|2 = 0.368 / 0.3681 = 1.000
  - p_3|2 = 0.0001 / 0.3681 = 0.000

* From x3's perspective:
  - exp(-16.0) = 0.000
  - exp(-9.0) = 0.0001
  - Sum = 0.0001
  - p_1|3 = 0.000 / 0.0001 = 0.000
  - p_2|3 = 0.0001 / 0.0001 = 1.000

---

## Step 3: Symmetrize Probabilities (p_ij)
Formula: p_ij = (p_j|i + p_i|j) / (2 * n), where n = 3

* p_12 = (1.000 + 1.000) / 6 = 0.333  <-- Target similarity
* p_13 = (0.000 + 0.000) / 6 = 0.000  <-- Target similarity
* p_23 = (0.000 + 1.000) / 6 = 0.167  <-- Target similarity

---

## Step 4: Low-Dimensional Map Initialization
Random initial coordinates in 2D space:
* y1 = [0.1, 0.2]
* y2 = [0.5, 0.5]
* y3 = [0.2, 0.9]

---

## Step 5: Low-Dimensional Probabilities (q_ij)
Formula: w_ij = 1 / (1 + ||y_i - y_j||^2)  [Student-t Distribution]
         q_ij = w_ij / Sum_pairs(w)

* Calculate weights (w_ij):
  - ||y1 - y2||^2 = 0.25 -> w_12 = 1 / (1 + 0.25) = 0.800
  - ||y1 - y3||^2 = 0.50 -> w_13 = 1 / (1 + 0.50) = 0.667
  - ||y2 - y3||^2 = 0.25 -> w_23 = 1 / (1 + 0.25) = 0.800
  - Sum of all pair weights = 0.800 + 0.667 + 0.800 = 2.267

* Normalize weights to get probabilities (q_ij):
  - q_12 = 0.800 / 2.267 = 0.353  (Target p_12 = 0.333)
  - q_13 = 0.667 / 2.267 = 0.294  (Target p_13 = 0.000)
  - q_23 = 0.800 / 2.267 = 0.353  (Target p_23 = 0.167)

---

## Step 6: Gradient Descent Optimization
Forces are calculated based on the difference between q_ij and p_ij:
* q_12 (0.353) approx p_12 (0.333) -> Small force between y1 and y2.
* q_13 (0.294) >> p_13 (0.000)      -> Strong repulsive force to push y1 and y3 apart.
* q_23 (0.353) > p_23 (0.167)       -> Moderate repulsive force to push y2 and y3 apart.

Coordinates y1, y2, y3 are updated. Steps 5 and 6 repeat for 500-1000 iterations.


### ⚪ Why a t-distribution? The crowding problem

This is the question that separates people who've read the paper from people who've read a blog post.

In high dimensions there is vastly more room at moderate distances than in 2-D — the volume of a shell grows like `r^(d−1)`, so a point can have many neighbours at "medium" distance. Squeeze that into a plane using a Gaussian in both spaces and all those moderately-distant points get crushed on top of each other. That's the **crowding problem**.

The Student-t has **heavy tails**, so a moderate distance in high-D can be represented by a much larger distance in 2-D at little cost. The effect is that clusters push apart and become visually distinct instead of collapsing into one blob.

### ⚪ Why it preserves local but not global structure

Because **KL divergence is asymmetric**. The cost is `Σ pᵢⱼ log(pᵢⱼ/qᵢⱼ)`:

- `p` large (close in high-D), `q` small (far in the plot) → `log` of a big ratio, weighted by a big `p` → **enormous penalty**
- `p` small (far in high-D), `q` large (close in the plot) → weighted by a tiny `p` → **negligible penalty**

So t-SNE is punished severely for tearing neighbours apart, and barely punished for placing distant points near each other. **That asymmetry is precisely why you can trust local neighbourhoods and must not trust the global layout.** Everything in §7.3 follows from this one fact.

---

## 7.3 🔴 The caveats — the part interviewers test

I ran this to make the point concrete. Three clusters in 50 dimensions, deliberately built with different spreads and separations:

```
                          within-cluster spread      centroid distances
ORIGINAL 50-D            A=3.5   B=34.9   C=3.5      A–B=212    B–C=1909   (ratio 9.0)
AFTER t-SNE              A=3.1   B=3.1    C=3.1      A–B=34.6   B–C=36.4   (ratio 1.05)
```

Cluster B was **ten times wider** than A and C. After t-SNE all three are identical in size. Cluster C was **nine times further** from B than A was. After t-SNE they're equidistant.

**So, concretely:**

**① Cluster sizes mean nothing.** t-SNE expands dense clusters and contracts sparse ones — that's the per-point bandwidth `σᵢ` doing its job. A big blob is not a more variable group.

**② Distances between clusters mean nothing.** Two clusters far apart on the plot may not be far apart in the data. You cannot say "group A is more similar to B than to C" from a t-SNE plot.

**③ Perplexity changes the picture substantially.** Same data, same seed:
```
perplexity =   5   → within-cluster spread ≈ 13.9
perplexity =  30   → within-cluster spread ≈  3.0
perplexity = 100   → within-cluster spread ≈  0.9
```
Always run 3–4 values (5–50 is the usual range) before believing anything.

**④ It's stochastic.** Different seeds, different plots. Fix `random_state`. *(sklearn now defaults to `init='pca'`, which is more stable and reproducible than the old random init — worth knowing.)*

**⑤ It can manufacture clusters that don't exist.** Run it on pure random noise with low perplexity and you will see convincing-looking clumps. **Never conclude "there are 5 groups" from a t-SNE plot alone** — verify with a clustering algorithm on the original data, silhouette scores, or a known label.

**⑥ There is no `transform()`.** t-SNE optimises the positions of *these specific points*; it doesn't learn a reusable mapping. `sklearn.manifold.TSNE` has `fit_transform` but genuinely no `transform` — you can check with `hasattr(TSNE, 'transform')` → `False`. That's why it can't go in a production pipeline.

**⑦ Don't cluster on t-SNE output.** Since distances are distorted, k-means on a 2-D t-SNE embedding is clustering an artefact. Cluster in the original space (or on PCA output) and use t-SNE to *display* the result.

> **The line to say:** *"t-SNE is a visualisation tool. It preserves local neighbourhoods, so points near each other on the plot were near each other in the data — but cluster sizes, the gaps between clusters, and the number of apparent clusters are all artefacts. I'd use it to look at data, never as a preprocessing step, and I'd verify any clusters I thought I saw against the original space."*

---

## 7.4 🟡 t-SNE hyperparameters

| Parameter | What it does | Guidance |
|---|---|---|
| **`perplexity`** | Roughly *"how many neighbours count as local"* — sets the per-point Gaussian bandwidth | Default 30; try 5–50. Must be **< n_samples**. Low → many small fragmented clusters; high → everything merges |
| `n_iter` / `max_iter` | Gradient descent steps | ≥ 1000. If the plot looks like a compressed ball, it hasn't converged |
| `learning_rate` | Step size | `'auto'` (≈ n/12). Too low → dense ball; too high → scattered noise |
| `init` | Starting layout | **`'pca'`** — now the sklearn default; more stable and reproducible than `'random'` |
| `early_exaggeration` | Inflates `P` early so clusters separate before fine-tuning | Rarely needs changing (default 12) |
| `metric` | Distance measure | `'euclidean'`; use `'cosine'` for text embeddings |

**Complexity:** naive `O(n²)`; the Barnes-Hut approximation (sklearn's default for `n_components < 4`) gets it to `O(n log n)`. Still slow beyond ~10k points — which is a large part of why UMAP took over.

---

## 7.5 🔴 UMAP — and how it differs

**Uniform Manifold Approximation and Projection** (McInnes, Healy & Melville, 2018). The theory is built on Riemannian geometry and fuzzy simplicial sets, but the practical algorithm is two steps:

1. **Build a weighted k-nearest-neighbour graph.** Edge weights encode how confident we are that two points are neighbours, with each point's local metric normalised by its distance to its nearest neighbour — so, like t-SNE's per-point bandwidth, it adapts to varying density.
2. **Optimise a low-dimensional layout** so its own k-NN graph matches, minimising a **cross-entropy** loss via stochastic gradient descent with negative sampling (the same trick word2vec uses).

**The key difference from t-SNE:** cross-entropy has a meaningful **repulsive term for non-neighbours**, where t-SNE's KL divergence effectively doesn't. UMAP is therefore penalised for placing unrelated points close together — which is why it retains noticeably more global structure.

### Why UMAP is usually the better default

| | t-SNE | UMAP |
|---|---|---|
| Speed | Slow; painful past ~10k points | **Much faster**; handles millions |
| Global structure | Poor | **Better** (still not fully trustworthy) |
| `transform()` for new data | **No** | **Yes** — can sit in a pipeline |
| Embedding dimensions | Really only 2–3 | Works for higher `k`, so usable as actual DR |
| Supervised mode | No | **Yes** — can use labels to guide the embedding |
| Deterministic | No | No (but `random_state` works) |

### UMAP's hyperparameters

| Parameter | What it does | Guidance |
|---|---|---|
| **`n_neighbors`** | Local vs. global balance — the analogue of perplexity | Default 15. Small (2–5) → fine local detail, fragmented. Large (50–200) → broad global structure |
| **`min_dist`** | How tightly points may pack in the embedding | Default 0.1. Small (0.0–0.1) → tight clumps, good if you'll cluster afterwards. Large (0.5–0.99) → evenly spread, better for seeing overall topology. **Purely about the picture — it has no effect on the underlying structure** |
| `n_components` | Output dimensions | 2 for plots; 10–50 if using it as real DR |
| `metric` | Distance | `'euclidean'`, `'cosine'` for embeddings, `'hamming'` for binary |

⚠️ **UMAP is not caveat-free.** It's still stochastic, still distorts distances, and can still invent apparent clusters. It *preserves more* global structure than t-SNE, not *all* of it. Cluster sizes remain unreliable. Most of §7.3 still applies — just less severely.

---

## 7.6 🔴 The standard workflow

> **PCA first (to ~50 dimensions), then t-SNE or UMAP.**

This is the recipe practitioners actually use, and saying it unprompted signals real experience. Three reasons:

1. **Speed.** Both methods are dominated by neighbour computations, which are much cheaper in 50 dimensions than in 5,000.
2. **Denoising.** PCA strips low-variance directions, which are largely noise; the neighbour graph then reflects real structure.
3. **Distance quality.** Euclidean distance degrades in very high dimensions, and both methods depend entirely on neighbour distances being meaningful.

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)          # 1. scale
X_pca    = PCA(n_components=50, random_state=0).fit_transform(X_scaled)   # 2. denoise
X_emb    = TSNE(n_components=2, perplexity=30,
                init='pca', random_state=0).fit_transform(X_pca)          # 3. visualise

# UMAP equivalent (pip install umap-learn)
# import umap
# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
# X_emb   = reducer.fit_transform(X_pca)
# X_new_emb = reducer.transform(X_new)      # ← t-SNE cannot do this
```

⚠️ **Scale before PCA**, for all the reasons in §4.5 — and both t-SNE and UMAP are distance-based, so unscaled features distort the neighbour graph too.

---

## 7.7 🟡 Choosing between them

```
Need it in a production pipeline / for a downstream model?
    → PCA  (deterministic, has transform, fast, interpretable variance)
    → UMAP only if the structure is genuinely non-linear AND you accept stochasticity

Just want to LOOK at the data?
    → UMAP  (faster, better global structure, transform available)
    → t-SNE if you specifically want the cleanest local cluster separation

Have labels and want separation?
    → LDA (supervised, linear) or supervised UMAP

Sparse text / TF-IDF?
    → TruncatedSVD (→ LSA), optionally UMAP on top for the picture

Need to explain it to a regulator?
    → None of these. Feature selection.
```

**A one-line summary worth having:** *"PCA for anything that needs to be reproducible and go into a model; UMAP for looking at data; t-SNE when I want the cleanest local cluster picture and I have time to run it."*

---

## 7.8 🔴 Interview Q&A

**Q: What's the difference between PCA and t-SNE?**
> "PCA is linear, deterministic, and preserves global structure — large distances stay large — and it gives you a reusable transform you can apply to new data. t-SNE is non-linear and stochastic, and it optimises local neighbourhood preservation at the explicit expense of global structure. Practically: PCA is a preprocessing step that also happens to be useful for plots; t-SNE is a plotting tool that should never be a preprocessing step. It has no transform method at all, so it can't go in a pipeline."

**Q: Your t-SNE plot shows five clean clusters. Can you conclude there are five groups?**
> "No, and I'd be careful about that one — t-SNE can produce convincing-looking clusters from pure noise, especially at low perplexity. Before believing it I'd re-run at several perplexity values and several seeds to see whether five is stable, then verify in the original space: cluster there and check silhouette scores, or check whether the groups correspond to a known label. The plot is a hypothesis, not evidence."

**Q: In a t-SNE plot, cluster A is much bigger than cluster B, and it sits far from cluster C. What can you conclude?**
> "Essentially nothing from either observation. Cluster size is an artefact — t-SNE adapts its bandwidth to local density, so it expands tight clusters and contracts diffuse ones. In a test I ran, a cluster genuinely ten times wider than its neighbours came out exactly the same size after t-SNE. And between-cluster distance is equally unreliable: two clusters nine times further apart than another pair came out equidistant. The only thing I'd trust is that points plotted near each other were genuinely near each other in the data."

**Q: What is perplexity?**
> "Roughly the effective number of neighbours each point considers — it sets the bandwidth of the per-point Gaussian used to convert distances into probabilities. Because the bandwidth is fitted per point to hit that target entropy, the notion of 'neighbourhood' adapts to local density automatically. Typical values are 5 to 50, and it must be less than the sample size. Low perplexity fragments the data into many small clusters, high perplexity merges everything, and the picture changes enough between settings that I'd always try several."

**Q: Why does t-SNE use a t-distribution rather than a Gaussian in the low-dimensional space?**
> "The crowding problem. High-dimensional space has far more room at moderate distances than a plane does, so if you use a Gaussian in both spaces all the moderately-distant points get crushed together in the embedding. The t-distribution's heavy tails let a moderate high-dimensional distance map to a much larger low-dimensional one cheaply, so clusters spread out and become visually distinguishable."

**Q: Why can't you apply t-SNE to a test set?**
> "Because it doesn't learn a mapping — it directly optimises the coordinates of the specific points you gave it, by gradient descent on a loss defined over those points' pairwise similarities. There's no function to apply to a new point. sklearn's TSNE genuinely has no `transform` method. UMAP does, because it learns the embedding as a parametrised optimisation over a neighbour graph it can extend to new points, which is one of the practical reasons to prefer it."

**Q: When would you use UMAP over t-SNE?**
> "Most of the time, honestly. It's substantially faster and scales to far more points, it preserves more global structure because its cross-entropy loss includes a real repulsive term for non-neighbours where t-SNE's KL divergence effectively doesn't, and it has a `transform` so it can handle new data. It also supports a supervised mode. I'd still reach for t-SNE if I specifically wanted the cleanest local cluster separation for a figure. And I'd be clear that UMAP is *better* on global structure, not *reliable* on it — most of the same caveats apply, just less severely."

**Q: You need to reduce 5,000 features to 50 for a downstream classifier. t-SNE or PCA?**
> "PCA, without hesitation. t-SNE is for visualisation — it has no transform, so I couldn't apply the same reduction to test data or to anything in production; it's stochastic, so the features would change between runs; it's very slow in more than 2–3 output dimensions; and its distance distortions would actively mislead a distance-based classifier. PCA is deterministic, fast, reversible, and gives me an explained-variance curve to pick 50 defensibly. If I suspected genuinely non-linear structure I'd consider UMAP or an autoencoder as an alternative, but I'd validate that it actually beat PCA on the downstream metric before accepting the extra complexity and non-determinism."

**Q: How would you tune t-SNE or UMAP? There's no loss to cross-validate.**
> "There's no supervised objective, so it isn't tuning in the usual sense — it's a robustness check. I'd sweep the main parameter (perplexity, or `n_neighbors`) across a few values and several seeds, and look for structure that *persists*. Anything that appears at one setting and vanishes at the next, I don't report. If I have labels I can check whether known classes separate, and there are quantitative neighbourhood-preservation measures like trustworthiness and continuity — sklearn has `trustworthiness` — but in practice the honest answer is that the plot is exploratory and any finding needs confirming in the original space."


---

# Part 8 — Method comparison

## 8.1 The landscape

**Linear methods:** PCA, SVD, Factor Analysis, LDA (Linear Discriminant Analysis), ICA
**Non-linear methods:** t-SNE, UMAP, Kernel PCA, Autoencoders, Isomap

Also worth distinguishing:
- **Feature *extraction*** (PCA, SVD) — builds *new* features from combinations of old ones. Loses interpretability.
- **Feature *selection*** (L1, mutual information, RFE) — keeps a *subset* of original features. Preserves interpretability.

> If an interviewer asks "how would you reduce dimensionality?", saying **"first I'd ask whether we need extraction or selection, because it depends on whether interpretability is a requirement"** is a better opening than jumping to PCA.

## 8.2 Comparison table

| Method | Linear? | Supervised? | Preserves | Main use | Watch out |
|---|---|---|---|---|---|
| **PCA** | ✅ | ❌ | Global variance | General DR, decorrelation | Scale-sensitive; linear only |
| **SVD / TruncatedSVD** | ✅ | ❌ | Global structure | Sparse text, recommenders | No centring in Truncated |
| **LDA** *(Linear Discriminant Analysis)* | ✅ | ✅ | Class separability | DR when you have labels | Max `C−1` components |
| **ICA** | ✅ | ❌ | Statistical independence | Blind source separation (EEG, audio) | Needs non-Gaussian sources |
| **Kernel PCA** | ❌ | ❌ | Non-linear structure | Curved manifolds | Kernel choice; scales poorly |
| **t-SNE** | ❌ | ❌ | **Local** neighbourhoods | **Visualisation only** | See warnings below |
| **UMAP** | ❌ | ❌ | Local + some global | Visualisation, sometimes DR | Stochastic; hyperparameter-sensitive |
| **Autoencoder** | ❌ | ❌ | Whatever it learns | Complex non-linear DR | Needs lots of data; a linear AE with MSE loss ≈ PCA |

⚠️ **The LDA name collision** — a classic interview trip-up:
- **Linear Discriminant Analysis** — *supervised* dimensionality reduction, maximises between-class separation over within-class scatter
- **Latent Dirichlet Allocation** — *unsupervised* topic modelling for text

They're unrelated. Ask which one is meant if it's ambiguous; it makes you look careful, not confused.

## 8.3 t-SNE and UMAP

Covered in depth in **Part 7**. The one-line version: both are **visualisation tools**, both are stochastic, and cluster sizes and between-cluster distances on their plots are artefacts. UMAP is the better default — faster, more global structure retained, and it has a `transform()` so it can handle new data.

# Part 9 — Interview question bank

## 9.1 Conceptual

**Q: Explain PCA to a non-technical stakeholder.**
> "Imagine you have 50 measurements about each customer, and many of them are telling you the same thing in different ways — annual income, monthly income, and spending limit all move together. PCA finds a smaller set of new summary scores that capture most of what varies across customers. Instead of 50 columns you might keep 6, lose very little information, and get a model that trains faster and is less likely to be thrown off by redundancy."

**Q: Explain PCA technically, in 60 seconds.**
> "PCA finds a new set of orthogonal axes ordered by how much variance in the data they capture. Mechanically: standardise the features, compute the covariance matrix, take its eigenvectors and eigenvalues, sort by eigenvalue descending, and project the data onto the top k eigenvectors. The eigenvectors are the directions — the principal components — and the eigenvalues tell you how much variance each one carries. Because the covariance matrix is symmetric, the eigenvectors are guaranteed real and mutually orthogonal, which is why the resulting components are uncorrelated."

**Q: What are eigenvalues and eigenvectors, intuitively?**
> "When a matrix acts on a vector it usually rotates and stretches it. Eigenvectors are the special directions that don't rotate — the transformation only stretches or shrinks them. The eigenvalue is that stretch factor. For PCA, the covariance matrix's eigenvectors are the natural axes of the data cloud, and the eigenvalues are the variance along each."

**Q: Why must eigenvectors come from a square matrix?**
> "Because `Av = λv` requires the output to live in the same space as the input, so `A` has to map a space to itself. For rectangular matrices the analogous concept is singular vectors, which is what SVD gives you — and that's part of why SVD is more general."

**Q: Why are principal components orthogonal?**
> "Two reasons that agree. Mathematically, the covariance matrix is symmetric, and the eigenvectors of a symmetric matrix are guaranteed mutually orthogonal. Conceptually, the constraint exists so each successive component captures *new* variance — if PC2 weren't orthogonal to PC1, it would be partly re-describing information PC1 already captured. Orthogonality is also what makes the resulting components uncorrelated, which is the point when you're fixing multicollinearity."

**Q: Do you need to scale the data before PCA?**
> *(See §4.5 — the single most likely follow-up.)* "Yes, when features are on different scales. PCA maximises variance and variance is scale-dependent, so an income feature in dollars will dominate PC1 over an age feature simply because dollars are numerically larger. Standardising first is equivalent to running PCA on the correlation matrix rather than the covariance matrix. The exception is when all features are already in the same units and comparable — image pixels, say — where variance differences are real signal rather than a unit artefact."

**Q: How do you choose the number of components?**
> *(See §4.7.)* "Cumulative explained variance as a starting point, a scree plot to eyeball the elbow, but ideally I'd treat k as a hyperparameter and cross-validate the downstream model. 95% of variance isn't inherently the right target — variance isn't the same thing as predictive signal."

**Q: What's the relationship between PCA and SVD?**
> *(See §5.3.)* "They give the same answer. If you run SVD on the mean-centred data matrix `X = USVᵀ`, then `V` holds the principal components and the eigenvalues of the covariance matrix are `s²/(n−1)`. sklearn's PCA is implemented via SVD because forming `XᵀX` squares the condition number and hurts numerical stability, and because SVD is cheaper when you have far more features than samples."

**Q: When would PCA fail or be inappropriate?**
> "Five cases. Non-linear structure — PCA only finds linear directions, so a spiral or Swiss-roll manifold needs kernel PCA or UMAP. When low-variance directions carry the signal — PCA is unsupervised, it never sees `y`, so it can discard exactly the feature that separates your classes. Categorical or one-hot data, where variance isn't meaningful. Sparse high-dimensional data like TF-IDF, where centring destroys sparsity — use TruncatedSVD. And anywhere interpretability is a hard requirement, like credit decisioning, because you can't explain a rejection in terms of PC3."

**Q: Does PCA always improve model performance?**
> "No, and I'd be cautious about the assumption. It reliably helps when multicollinearity is hurting a linear model, when you're compute-constrained, or when `d` is large relative to `n`. It can hurt when discriminative signal sits in a low-variance direction, and it's often unnecessary for tree-based models, which handle correlated features and irrelevant features reasonably well. I'd treat it as a hypothesis to test with cross-validation, not a default preprocessing step."

**Q: Is PCA supervised or unsupervised?**
> "Unsupervised — it never looks at the target. That's exactly its limitation for predictive modelling, and the reason **Linear Discriminant Analysis** exists as the supervised alternative: LDA maximises class separability rather than raw variance."

**Q: What does a negative loading mean?**
> "That the feature moves inversely along that component. But note the sign of an eigenvector is arbitrary — flipping every loading in a component gives an equally valid component. Only the *relative* signs within a component are meaningful."

**Q: Why does PCA use eigenvectors? Where does that come from?**
> *(Full derivation in §4.9.)* "It falls out of the optimisation. You want the unit vector `w` maximising the variance of the projected data, which is `wᵀΣw` subject to `wᵀw = 1`. Setting up the Lagrangian and differentiating gives `Σw = λw` — the eigenvector equation. And substituting back shows the variance along that direction equals λ itself, so maximising variance means taking the largest eigenvalue. That single derivation explains why eigenvectors, why they're sorted by eigenvalue, and why explained variance is λᵢ/Σλ."

**Q: Does PCA remove multicollinearity?**
> "Completely — the components are uncorrelated by construction, since `WᵀΣW = Λ` is diagonal. Running a regression on the PC scores is called **Principal Component Regression**. The catch is that PCA chooses components by variance without ever looking at `y`, so PCR can throw away a low-variance direction that was highly predictive. **Partial Least Squares** fixes that by choosing directions maximising covariance *with the target*."

**Q: What's the computational complexity of PCA?**
> "Forming the covariance matrix is `O(nd²)` and eigendecomposing it is `O(d³)`. That `d³` term is what makes the covariance route impractical for large feature counts — one of the reasons sklearn uses SVD instead. If you only need the top `k` components, randomised SVD gets you to roughly `O(ndk)`."

**Q: How do you handle missing values before PCA?**
> "PCA has no native handling and sklearn will error. I'd impute first — median, or iterative/MICE if the missingness is structured — or use probabilistic PCA with EM, which handles it in a principled way rather than fabricating values. What I wouldn't do is fill with zeros on unstandardised data: zero isn't neutral, it's an extreme value that will drag the components toward it."

**Q: What if the features are already uncorrelated?**
> "Then PCA does essentially nothing useful. The covariance matrix is diagonal, its eigenvectors are the original coordinate axes, and PCA just reorders your columns by variance. Worth checking the correlation matrix before reaching for it — if there's no correlation structure, there's no redundancy to exploit."

**Q: How would you make PCA components interpretable?**
> "**Sparse PCA** — it adds an L1 penalty to the loadings so most are driven to zero, meaning each component involves only a handful of original features and you can actually name it. You trade some explained variance for interpretability. Alternatively, drop feature extraction entirely and use feature *selection* — L1 regularisation or mutual information — which keeps the original variables intact. In a regulated setting I'd usually go the second route."

**Q: Can PCA be used for outlier detection?**
> "Yes, in two ways. Project the data down to k components, reconstruct, and measure reconstruction error — points that reconstruct poorly don't fit the dominant structure. Or look at the scores on the trailing, low-variance components, where outliers often show up. The caveat is that outliers also *distort* the components in the first place, since variance uses squared deviations — so a robust PCA variant is often the safer tool."

## 9.2 Mathematical

**Q: Compute the eigenvalues of `[[1,4],[3,2]]`.**
> Full working in §2.4. `det(A−λI) = (1−λ)(2−λ) − 12 = λ² − 3λ − 10 = (λ−5)(λ+2)` → `λ = 5, −2`. Quick check: trace = 3 = 5 + (−2) ✓, det = −10 = 5 × (−2) ✓.

**Q: Find the eigenvalues of `[[7,2],[2,4]]`.**
> Don't reach for the quadratic formula. `trace = 11`, `det = 28 − 4 = 24`. Two numbers adding to 11 and multiplying to 24 → **λ = 8 and 3**. Explained variance 73% / 27%. *(§4.4)*

**Q: What are the eigenvalues of `[[9,0],[0,4]]`, and what does it tell you?**
> "λ = 9 and 4 — for a diagonal matrix the eigenvalues *are* the diagonal entries, and the eigenvectors are the original axes. More importantly, zero covariance means the features are already uncorrelated, so PCA has nothing to rotate. It would just relabel my existing features in variance order, which means PCA isn't buying me anything here." *(§4.4)*

**Q: Given explained variances 40/25/15/10/5/5%, how many components to reach ≥80%?**
> `40 + 25 + 15 = 80%` → **3 components**.

**Q: SVD on a 1000×500 ratings matrix. The first 50 singular values give 85% of variance and you need 90%. What do you do?**
> **Increase to roughly 60 components.** Not 50 — 85% doesn't meet the stated requirement, and "close enough" isn't a decision you get to make when the threshold was specified. Not 100 either — that's over-shooting, discarding the compression benefit and risking retaining noise components. The right move is to compute the cumulative explained variance curve and take the smallest k crossing 90%.

**Q: What's the shape of each matrix in `M = UΣVᵀ` for an `m×n` matrix?**
> Full SVD: `U` is `m×m`, `Σ` is `m×n`, `Vᵀ` is `n×n`. Reduced/thin SVD (`full_matrices=False`): `U` is `m×r`, `Σ` is `r×r`, `Vᵀ` is `r×n` with `r = min(m,n)`.

**Q: If a covariance matrix has an eigenvalue of 0, what does that tell you?**
> "The matrix is singular and rank-deficient — at least one feature is an exact linear combination of the others. Perfect multicollinearity. The data actually lives in a lower-dimensional subspace, so that direction carries zero variance and can be dropped with no loss at all."

**Q: Can eigenvalues be negative? Can singular values?**
> "Eigenvalues of a general matrix, yes — `[[1,4],[3,2]]` has `λ = −2`, meaning that direction gets flipped. But eigenvalues of a **covariance** matrix are always ≥ 0, because it's positive semi-definite and they represent variances. **Singular values are always ≥ 0 by definition** — they're square roots of the eigenvalues of `MᵀM`, which is always PSD."

**Q: Prove the principal components are uncorrelated.**
> "The covariance of the projected scores is `Wᵀ Σ W`. Substituting the eigendecomposition `Σ = WΛWᵀ` gives `Wᵀ W Λ Wᵀ W = Λ`, since `W` is orthogonal so `WᵀW = I`. `Λ` is diagonal, so all off-diagonal covariances are zero — the components are uncorrelated by construction, and each PC's variance is its eigenvalue."

## 9.3 Applied / scenario

**Q: 10,000 features, 500 rows. How do you approach it?**
> "The `d ≫ n` regime, so the covariance matrix is 10,000×10,000 and rank-deficient — at most 499 non-zero eigenvalues. I'd use SVD rather than explicit eigendecomposition for exactly that reason. But before reaching for PCA I'd ask what the features are: if they're sparse text, TruncatedSVD; if many are near-constant or duplicated, cheap filtering first; if interpretability matters, regularised feature selection with L1 instead. I'd also be wary that with 500 rows any covariance estimate is noisy, so I'd validate carefully rather than trusting explained-variance numbers."

**Q: PCA improved training speed but hurt accuracy. Why?**
> "Most likely the discriminative signal sat in a low-variance direction that got discarded — PCA is unsupervised and optimises variance, not separability. Other candidates: I dropped too many components; I didn't standardise, so one high-variance feature dominated; or the model was tree-based, which doesn't benefit from decorrelation and does get hurt by losing the original axis-aligned splits. I'd check the explained-variance curve, sweep `k`, and compare against supervised alternatives like LDA or L1-based selection."

**Q: Your PCA loadings flipped sign after retraining. Is something broken?**
> "No. Eigenvectors are only defined up to a scalar multiple, including −1, so the sign is arbitrary and different solver runs or library versions can return either. What matters is the relative signs within a component and the explained-variance ratios. If it's causing downstream confusion I'd pin it by fixing `random_state` and applying a deterministic sign convention — for example forcing the largest-magnitude loading in each component to be positive."

**Q: Would you use PCA on a credit-risk model at a bank?**
> "Probably not for the production model, and I'd raise the reason early. Credit models need adverse-action reasons under ECOA/Reg B — you have to tell a declined applicant *why*. 'Your third principal component was too low' isn't an explainable reason, and model risk teams will push back on components with no business meaning. I'd use PCA for exploratory analysis and multicollinearity diagnostics, but for the model itself I'd prefer feature selection, which keeps the original interpretable variables."

**Q: How do you handle categorical features with PCA?**
> "PCA assumes continuous numeric input where variance is meaningful, and variance of a one-hot indicator isn't. Options: **MCA** (Multiple Correspondence Analysis), which is the categorical analogue; **FAMD** for mixed data; target/embedding encodings that produce genuinely continuous values; or simply don't reduce the categoricals and apply PCA only to the numeric block."

**Q: Users vs items — which similarity would you use for a retailer with 50M users and 10K products?**
> "Item–item. Two reasons. First the rule of thumb: when users vastly outnumber items, the item–item similarity matrix is 10K×10K rather than 50M×50M — orders of magnitude cheaper to compute and store. Second, item ratings are stable over time while user taste drifts, so the item similarity matrix can be recomputed nightly rather than constantly. That's essentially why Amazon's published algorithm is item–item."

**Q: A recommender's matrix is 99.5% empty. What do you do?**
> "First, don't fill the missing entries with zero — for explicit ratings that's asserting a rating that was never given, and the model learns your imputation instead of preference. I'd use matrix factorization that sums loss only over observed entries, solved by ALS or SGD, with regularisation because sparsity invites overfitting. If the signal is implicit — clicks and views rather than stars — then treating unobserved as weak negatives *is* appropriate, with a confidence-weighted objective. And I'd add a content-based fallback for cold-start users and items, since factorization has nothing to work with there."

## 9.4 Coding

**Q: Implement PCA from scratch.** — see §4.8.

**Q: Given a fitted `PCA`, which original features drive PC1?**
```python
import pandas as pd
loadings = pd.Series(pca.components_[0], index=feature_names)
loadings.abs().sort_values(ascending=False).head(10)
```

**Q: Recover the original data from the components.**
```python
X_approx = pca.inverse_transform(X_reduced)
mse = ((X_scaled - X_approx) ** 2).mean()
# exact only when n_components == n_features
```

**Q: Compress an image with SVD.**
```python
U, S, Vt = np.linalg.svd(img_gray, full_matrices=False)
k = 50
img_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
compression = k * (img_gray.shape[0] + img_gray.shape[1] + 1) / img_gray.size
```

---

# Part 10 — ⚠️ Errata in the source material

**Read this before revising from the original slides.** Everything below was verified numerically.

## 10.1 The covariance worked example has arithmetic errors

For the height/weight/age/income table (§3.4), using divisor `n = 3`:

| Quantity | Deck says | Correct | Error |
|---|---|---|---|
| `Cov(height, age)` | 66.67 | **33.33** | 2× |
| `Cov(height, income)` | 133,333.33 | **13,333.33** | 10× |
| `Var(income)` | 266,666,666.67 | **2,666,666.67** | 100× |

```python
X = np.array([[160,55,25,3000],[170,65,30,5000],[180,75,35,7000]], float)
np.cov(X, rowvar=False, bias=True)
# [[     66.67,     66.67,     33.33,  13333.33],
#  [     66.67,     66.67,     33.33,  13333.33],
#  [     33.33,     33.33,     16.67,   6666.67],
#  [  13333.33,  13333.33,   6666.67, 2666666.67]]
```

Also worth knowing: **that example dataset is rank 1** — all four variables are perfectly collinear, so its covariance matrix has exactly one non-zero eigenvalue. It's a valid arithmetic drill but a degenerate statistics example.

## 10.2 The eigenvector equation is written backwards

The deck shows `v⃗(A − λI) = 0`. It should be **`(A − λI)v⃗ = 0`** — with `v` as a column vector, the dimensions only work with the matrix on the left. The subsequent `Det(A − λI) = 0` is correct.

## 10.3 SVD matrix dimensions are inconsistent

The text says `U` is `m×n` and `D` is `n×n`; the diagram on the same slide shows `U` as `m×m` and `Σ` as `m×n`. **The diagram is right** for the full SVD. Reduced SVD gives `U` as `m×r`, `Σ` as `r×r`, `r = min(m,n)`.

## 10.4 "Σ has eigenvalues in increasing order"

From the personal notes. Two corrections: they are **singular values**, not eigenvalues, and they are sorted **decreasing** — which is precisely what makes "keep the top k" meaningful.

## 10.5 The PCA standardisation inconsistency

The deck's PCA steps list standardisation as step 1, but the baseball covariance matrix shown (`[[2503.33, 1596.74],[1596.74, 1061.01]]`) is computed on **raw, unstandardised** data — standardised data would give a correlation matrix with 1s on the diagonal. Here it barely matters (99.15% vs 98.99% for PC1 because both features are counts on a similar scale), but don't reproduce the inconsistency.

## 10.6 PC2's sign is inconsistent across slides

One slide gives `PC2 = 0.54·hit − 0.84·single`; a later one gives `PC2 = −0.54·hit + 0.84·single`. **Both are correct** — eigenvector signs are arbitrary. Just don't let it confuse you.

## 10.7 "Fill the null values with 0"

Stated for the recommender ratings matrix. For **explicit** ratings this is the wrong default — it asserts a rating the user never gave. Proper matrix factorization sums the loss only over observed entries. For **implicit** feedback, zero-filling with confidence weighting is legitimate. See §6.5.

---

# Part 11 — Cheat sheet

```
════════ EIGEN ════════
Av = λv                     v = direction unchanged, λ = stretch factor
(A − λI)v = 0               → det(A − λI) = 0  [characteristic equation]
Σλ = trace(A)               Πλ = det(A)          ← fast sanity checks
Square matrices only.  Symmetric ⇒ real λ, orthogonal v.
Eigenvectors are unique only up to scale — SIGN IS ARBITRARY.

════════ COVARIANCE ════════
Cov(X,Y) = Σ(xᵢ−x̄)(yᵢ−ȳ)/(n−1)      Cov(X,X) = Var(X)
Covariance matrix: symmetric, PSD, diagonal = variances
CORRELATION MATRIX = COVARIANCE MATRIX OF STANDARDISED DATA   ← remember this

════════ PCA ════════
1 standardise → 2 covariance matrix → 3 eigendecompose
4 sort by λ desc → 5 loadings = top eigenvectors
6 explained var = λᵢ/Σλ → 7 repeat → 8 project X_c · W_k

WHY EIGENVECTORS:  maximise wᵀΣw s.t. wᵀw=1  → Lagrangian → Σw = λw
                   variance along w = λ  ⟹  biggest λ = PC1
PC1 = direction of max variance
PC2 = max remaining variance, ORTHOGONAL to PC1
Components are UNCORRELATED. d features → up to d components.

⚠️ SCALE FIRST. Variance is scale-dependent; biggest units win PC1.
⚠️ UNSUPERVISED — never sees y. Low-variance ≠ low-signal.
⚠️ LINEAR only.   ⚠️ Components are NOT interpretable.
⚠️ Fit scaler + PCA on TRAIN only (use a Pipeline).
np.linalg.eigh (not eig) for covariance; returns ASCENDING — reverse it.

════════ SVD ════════
M = U Σ Vᵀ    any m×n matrix (no square/symmetry requirement)
U: m×m left singular vectors  (eigenvectors of MMᵀ)
Σ: m×n singular values, DESCENDING  (= √eigenvalues of MᵀM)
Vᵀ: n×n right singular vectors (eigenvectors of MᵀM)

PCA ↔ SVD on centred X:
   components  = V
   λᵢ          = sᵢ²/(n−1)
   sᵢ          = √(λᵢ(n−1))
   scores      = U·S
SVD is used because: numerically stabler (no XᵀX), better when d≫n, works on
non-square matrices.  PCA centres; TruncatedSVD does NOT → use it on sparse.
TruncatedSVD on TF-IDF = LSA.  Eckart–Young: best rank-k approximation.

════════ MATRIX FACTORIZATION ════════
A(n×m) ≈ B(n×d) · Cᵀ(d×m)      Âᵢⱼ = bᵢᵀcⱼ
Minimise Σ over OBSERVED (Aᵢⱼ − bᵢᵀcⱼ)² + λ(‖B‖²+‖C‖²)    ← observed only!
Solve with SGD or ALS (ALS parallelises → Spark).
NMF: all entries ≥ 0 → additive parts → MORE INTERPRETABLE.
Item–item > user–user when users ≫ items and item ratings are stable.
Cold start → metadata / popularity / content-based fallback.

Complexity: O(nd²) form Σ + O(d³) eigendecompose;  randomised top-k ≈ O(ndk)
Variants: randomised (big data) · incremental (streaming) · sparse (interpretable)
          kernel (non-linear) · robust (outliers) · whiten=True (unit-variance PCs)
PCR = regression on PC scores (kills collinearity). PLS = supervised version.

════════ 2x2 EIGENVALUE SHORTCUTS (don't use the quadratic formula) ════════
① EQUAL DIAGONALS [[v,c],[c,v]] → λ = v+c, v−c ; eigvecs ALWAYS [1,1],[1,−1] (±45°)
     [1,1]↔v+c always.  c>0 → PC1=[1,1] (sum) ;  c<0 → PC1=[1,−1] (difference)
② ZERO OFF-DIAGONAL (diagonal Σ) → λ = the diagonal entries ; eigvecs = [1,0],[0,1]
     ⇒ features already uncorrelated ⇒ PCA only REORDERS columns, buys nothing
     (PC1 = axis with the LARGER diagonal entry — not necessarily the first!)
③ TRACE & DET: λ₁+λ₂ = trace, λ₁λ₂ = det → "what adds to T and multiplies to D?"
     [[7,2],[2,4]]: T=11, D=24 → 8 and 3.  Done.
   fallback: λ = ½[T ± √(T²−4D)]
ALWAYS verify: Σλ = trace ✓  Πλ = det ✓
Eigenvector: plug λ into (Σ−λI)v=0, read FIRST ROW ONLY (row 2 is redundant)

════════ WHITEBOARD EXAMPLE (memorise) ════════
x=[2,4,6,8,10] y=[2,6,4,10,8] → both mean 6
Σ = [[10,8],[8,10]]  → (10−λ)²=64 → λ = 18, 2   (trace 20 ✓ det 36 ✓)
v₁=[1,1]/√2  v₂=[1,−1]/√2    explained: 90% / 10%

════════ t-SNE / UMAP ════════
BOTH: non-linear, STOCHASTIC, VISUALISATION ONLY, preserve LOCAL neighbourhoods
t-SNE: distances→probabilities (Gaussian, per-point σ set by PERPLEXITY)
       low-dim uses STUDENT-t (heavy tails → fixes CROWDING problem)
       minimise KL(P‖Q) — ASYMMETRIC ⇒ punishes tearing neighbours apart,
       NOT punishes putting far points together ⇒ local ok, global meaningless
UMAP:  weighted k-NN graph → cross-entropy w/ negative sampling
       cross-entropy HAS a repulsive term ⇒ keeps MORE global structure
       n_neighbors (≈perplexity, local↔global) · min_dist (packing, cosmetic only)

⚠️ CLUSTER SIZES MEANINGLESS   ⚠️ GAPS BETWEEN CLUSTERS MEANINGLESS
⚠️ can invent clusters from noise   ⚠️ perplexity/seed change the picture
⚠️ t-SNE has NO transform() → can't go in a pipeline.  UMAP DOES.
⚠️ don't cluster on the embedding — cluster in the original space

WORKFLOW:  scale → PCA to ~50 dims → t-SNE/UMAP     (speed + denoise + better distances)
PICK:  pipeline/model → PCA.   Looking at data → UMAP.   Regulator → neither.

════════ CHOOSING k ════════
cumulative explained variance ≥ 90/95%  |  scree elbow
Kaiser λ>1 (correlation matrix only)    |  CV the downstream model ← best

════════ TOP 6 THINGS THEY ASK ════════
1 Explain PCA          2 Do you scale first? (YES — and why)
3 How to pick k        4 PCA vs SVD relationship
5 When does PCA fail   6 What do the components mean? (usually nothing)
```

---

## Further reading

- **Interactive PCA visualiser** — <https://setosa.io/ev/principal-component-analysis/> — the single best 10 minutes you can spend on PCA intuition
- **3Blue1Brown, *Essence of Linear Algebra*** — especially the eigenvector episode
- Jolliffe & Cadima (2016), *Principal component analysis: a review and recent developments*
- Koren, Bell & Volinsky (2009), *Matrix Factorization Techniques for Recommender Systems* — the Netflix Prize paper
- McInnes, Healy & Melville (2018), *UMAP: Uniform Manifold Approximation and Projection*
- van der Maaten & Hinton (2008), *Visualizing Data using t-SNE* — the original
- Wattenberg et al., *How to Use t-SNE Effectively* (Distill) — the caveats in §8.3, illustrated
