---
layout: distill
title: On the Complexity of Attention Variants (part i)
description: Analyzing NC1, TC0, and where different models fit. # todo: can have a catchier description / intro I think
tags: architecture, theory
date: 2025-10-01
published: true

authors:
  - name: Tarushii Goel
    url: "https://tarushii-g.github.io/"
    affiliations:
      name: MIT CSAIL
  - name: Adam Zweiger
    url: "https://adamzweiger.github.io/"
    affiliations:
      name: MIT CSAIL

bibliography: 2025-10-01-linear-attn-complexity.bib


---

Recent models such as [DeltaNet](https://sustcsonglin.github.io/assets/pdf/talk_250117.pdf) and [RWKV-7](https://arxiv.org/abs/2503.14456) make claims about moving beyond the $$\text{TC}^0$$ complexity class, which the transformer is confined to. Most discussion of neural network architecture complexity so far has remained in dense theoretical papers. This blog post will dig into what the claims actually mean from first principles, and why they may or may not matter in practice.

## Motivation: State Tracking

When we analyze the computational power of different architectures, it helps to start with concrete problems we care about. One very important class of problems is *state tracking*: given a sequence of inputs that update some underlying system, can the model keep track of the current state of that system?

A canonical example is group composition, specifically in the symmetric group on five elements, $S_5$. The problem is: given a sequence of permutations, compute their composition.

For example:
> Start with the ordering 1,2,3,4,5. Apply the following permutations: swap elements 1 and 3; rotate elements 1, 3, and 4; swap elements 4 and 5.

The final state is $4, 2, 3, 5, 1$. More generally, this is a version of the *group word problem*: given a sequence of group elements, is their product equal to the identity?

This type of problem implicitly shows up everywhere. Imagine trying to track the state of a chessboard as a sequence of moves is played, the state of a codebase as a sequence of diffs is applied, or the result of executing a program.

This particular problem is also quite significant from a complexity-theoretic perspective. The state-tracking problem for $S_5$ is known to be $\text{NC}^1$-complete, meaning it is "at least as hard" as any other problem in the complexity class $\text{NC}^1$, which we will later define. By contrast, problems in the (likely) weaker class $\text{TC}^0$ cannot simulate this task. As it turns out, an example of a problem that is in the weaker $\text{TC}^0$ class is *computing the result of a transformer's forward pass*. Hence, it can be shown that under the assumption that $\text{TC}^0\neq\text{NC}^1,$ transformers "cannot solve state-tracking."

The goal of this blog post is to prove this statement and similar statements for other architectures, give some intuition as to why certain architectures can or cannot track state, and analyze implications of this for designing expressive architectures.


## A Background on Circuit Complexity Classes

Our computation model is that we have a set of gates wired together that take in boolean inputs and compute some function of the inputs. Below is an example circuit with $$5$$ gates, $$3$$ inputs, and a depth of $$3$$.

{% include figure.liquid loading="eager" path="assets/img/1.webp" class="img-fluid rounded z-depth-1" %}

$\mathbf{NC}$ **(Nick's Class).**$\quad$  A circuit is in $$\text{NC}^k$$ if for $$n$$ input variables, it has depth at most $$O((\log{n})^k)$$. Here, we are allowed to use any 2-input or 1-input gate, such as AND, OR, XOR, NOR, etc. These are gates with bounded "fan-in" (number of inputs).

One example of a problem in $\text{NC}^1$ we gave is state-tracking on $S_5$. For illustrative purposes, let's start by looking at state-tracking on $S_2$, or computing the parity of $n$ bits. We can compute the parity of two bits through $x_i \oplus x_j$, and we can extend this to arbitrary sized inputs by constructing a binary tree:

<svg viewBox="0 0 720 280" width="100%" role="img" aria-label="Binary tree of XOR computing parity">
  <defs>
    <style>
      .node { fill: #ffffff; stroke: #333333; stroke-width: 2; }
      .gate { fill: #eef6ff; stroke: #1f6feb; stroke-width: 2; }
      .wire { stroke: #666666; stroke-width: 2; }
      .label { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
               Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', Arial, sans-serif;
               font-size: 14px; fill: #111111; }
      .caption { font-size: 13px; fill: #444444; }
    </style>
  </defs>

  <!-- Inputs -->
  <text class="label" x="80"  y="250">x₁</text>
  <text class="label" x="220" y="250">x₂</text>
  <text class="label" x="460" y="250">x₃</text>
  <text class="label" x="600" y="250">x₄</text>

  <!-- Bottom wires up to first XOR layer -->
  <line class="wire" x1="90"  y1="235" x2="150" y2="190" />
  <line class="wire" x1="230" y1="235" x2="170" y2="190" />
  <line class="wire" x1="470" y1="235" x2="530" y2="190" />
  <line class="wire" x1="610" y1="235" x2="550" y2="190" />

  <!-- First XOR layer (pairwise) -->
  <ellipse class="gate" cx="160" cy="180" rx="28" ry="18" />
  <text class="label" x="148" y="184">XOR</text>

  <ellipse class="gate" cx="540" cy="180" rx="28" ry="18" />
  <text class="label" x="528" y="184">XOR</text>

  <!-- Wires from first layer to top XOR -->
  <line class="wire" x1="188" y1="172" x2="350" y2="120" />
  <line class="wire" x1="512" y1="172" x2="370" y2="120" />

  <!-- Top XOR -->
  <ellipse class="gate" cx="360" cy="110" rx="28" ry="18" />
  <text class="label" x="348" y="114">XOR</text>

  <!-- Output wire and label -->
  <line class="wire" x1="360" y1="92" x2="360" y2="60" />
  <text class="label" x="320" y="48">x₁ ⊕ x₂ ⊕ x₃ ⊕ x₄</text>

  <!-- Caption -->
  <text class="label caption" x="360" y="275" text-anchor="middle">
    Binary XOR tree (depth O(log n)) computing parity via pairwise XOR
  </text>
</svg>

This binary tree decomposition is able to solve any word problem for groups (such as $S_5$) as long as the operation has constant depth, since group operations are associative, thus all such problems are in $\text{NC}^1$.

$\mathbf{TC}$ **(Threshold Circuit).**$\quad$  A circuit is in $$\text{TC}^0$$ if it has *constant depth* and a polynomial number of gates in the input. Constant depth is severely limiting, however, so we are additionally allowed to use a *threshold* gate in addition to the usual 2-input boolean ones. A threshold gate is defined as $$G(x_1, x_2, ..., x_n) = \textbf{1}_{\sum_i w_i x_i \geq \theta}$$ where the weights $$\textbf{w}$$ and threshold $$\theta$$ are parameters of the gate (rather than inputs). This is very powerful. For example, this gives us the ability to construct an AND with unbounded fan-in in a single gate by setting $$w_i = 1$$ and $$\theta = n$$. Thus, many of the circuits that would have required a logarithmic-depth binary tree can now be constructed in constant depth using threshold gates.

Despite the additional power that threshold gates provide, it is proven that $$\text{TC}^0 \subseteq \text{NC}^1$$. Whether this containment is strict or if it is an equality is still an open question. It is often assumed that the containment is strict, though [Ryan Williams](https://en.wikipedia.org/wiki/Ryan_Williams_(computer_scientist)) actually puts this at 50/50 odds. <d-cite key="williams2022estimatedlikelihoods"></d-cite>

Finally, many papers also refer to $$\textit{L}$$-*uniform* circuit families, which are circuits that can be generated by a deterministic Turing machine running in logarithmic space. This condition is often added to rule out hardwired circuit designs, but we will not concern ourselves with it here.

## Algorithms that are in $$TC_0$$

### Data Types

To understand the circuit complexity of different algorithms, we actually need to consider the datatype that we are performing arithmetic on. In practice, this is a fixed-size floating point or integer number. Adding fixed-size datatypes is clearly in $$TC_0$$, since the problem size is constant, but is also severely limiting of the theoretical expressivity of our model.

For example, a fixed precision transformer does not have the capability of attending to every KV pair uniformly. <d-cite key="merrill2022logprecisionlogic"></d-cite>
<!-- what does attending uniformly mean? -->

{% include dd_parent_open.liquid title="Proof" %}

In attention we have weights

\begin{align*}
a = \text{softmax}(q^T K), \;\; \sum_i a_i = 1 \;\; a_i \geq 0
\end{align*}

where each $a_i$ will be represented in $p$ bits for some fixed constant $p$. The smallest value that can be represented by a $p$-bit floating point number with $p_m$ mantissa bits and $p_e$ exponent bits is $2^{-(p_m + 2^{p_e - 1} - 2)}$, which is $\geq 2^{-2^{p}}$. Since $\sum_i a_i = 1$, only at most $2^{2^{p}}$ of the $a_i$ can be non-zero. This means that in theory attention at fixed precision can only attend to a fixed number of KV-pairs, behaving similary to hard attention. In practice, with $p=16$, the value $2^{2^p}$ is very large, so this is not a problem.

{% include dd_parent_close.liquid %}

This motivates using a *log-precision* transformer for our theoretical analysis, where the datatype is a $c \log{n}$-bit floating point number for $n$ tokens. This *more expressive* than the models we use in practice, but since we also work with small $n$ in practice, it shouldn't matter. We can also express a $c\log{n}$-bit floating point as an $O(n)$-bit integer, which means that circuits on the $O(n)$-bit integer should also work for the floating point representation. Thus, for simplicity, in the next few sections we will work with $O(n)$-bit integers. In reality, there are subtleties with this datatype switch, which you can read more about in Merrill et al. <d-cite key="merrill2024illusionstate"></d-cite>

### Addition

It's not immediately obvious how to add two numbers in constant depth since the algorithm we are taught in elementary school is inherently sequential: go right-to-left, adding each digit and computing a *carry* for the next column of digits. This is an $O(n)$ sequential algorithm for $n$-bit numbers. However, we *can* add two numbers in constant depth using [carry-lookahead addition](https://en.wikipedia.org/wiki/Carry-lookahead_adder). To add two numbers $a_n a_{n-1} .. a_{0}$ and $b_n b_{n-1} .. b_{0}$, we first compute the *propagator* bits $p_i$ and *generator* bits $g_i$. The generator bits indicate that the next most significant bit will have a carry of one *regardless* of the carry of the current bit, and the propagator bits indicate that the next bit will have a carry of $1$ if the current bit has a carry of $1$.

$$
p_i = a_i \lor b_i, \; \; g_i = a_i \land b_i
$$

$$
c_i = \textbf{1}_{\exists j \mid p_{i-1} \land p_{i-2}... \land p_{j+1} \land g_j}
$$

Thus computing the carry bit requires computing the propagators and generators, along with expressions of the form $p_{i-1} \land p_{i-2}... \land p_{j+1} \land g_j$. This expression is an AND with many inputs, which we've already shown can be computed with a single threshold gate! We can also compute an OR with unbounded fan-in by setting $\textbf{w} = 1$ and $\theta=1$, allowing us to check if least one AND expression is satisfied. With that, we can now add two numbers in $TC_{0}$!

In practice, it has been shown that one-layer transformers trained to perform $n$-digit addition do in fact learn an algorithm similar to a carry-lookahead adder, with parallel preparation for each digit, but with a fixed window instead of a full lookahead tree (hence there exist failures at edge cases). <d-cite key="quirke2024understandingadditiontransformers"></d-cite> Other work on modular addition discovered addition algorithms that are less carry-lookahead-like, though still involve parallel computation. <d-cite key="zhong2023clockpizzastoriesmechanistic"></d-cite>

### Iterated Addition

Now, let's look at the problem of adding $n$ numbers which are each $n$ bits.

We can start with the simpler problem of adding $n$ numbers which are each $m=1$ bits ($b_1, ... b_n$). First observe that we can construct a threshold for each of the $n$ possible values of the sum:
$$
\mathbf{1}_{\sum_j b_j \geq i}
$$. Then, we can construct a one-hot vector $\bf{o}$ for the value of the sum: $$o_i = \textbf{1}_{\sum_j b_j \geq i} \land \textbf{1}_{\sum_j -b_j \geq -i}$$. Then, this $n$-dimensional one-hot vector can be mapped to the exact output in a single layer of threshold circuits.

In fact, this approach works whenever the number of possible values of the sum is polynomial in $n$ (since we are only allowed a polynomial number of gates). The number of possible values of the sum is $n(2^m-1)+1$, so we can use the exact same approach to sum any polynomial number of $m=O(\log{n})$-bit numbers (for the threshold sums, we can weight a bit in position $i$ with weight $2^i$).

The last step is to extend this to $n$-bit numbers. Consider the $n \times n$ bit array that stacks all of the numbers on top of each other. We can organize these bits into groups of $\log{n}$ columns, and label each group as even or odd in a striped fashion. When we add a single even group, the result will have at most $2 \log{n}$ bits, which means that the it would not reach the next even group. Thus, we can add all of the bits in all of the even groups, and then add the odd groups, and issue a final addition to add them together.


Multiplication can be reformulated as iterated addition, which implies that multiplying two numbers is also in $TC_0$.


### Iterated Multiplication

The algorithm for iterated multiplication is slightly more complicated and relies on other results in circuit complexity theory, but it is also in $\text{TC}^0$. To give a rough outline of the construction, we take the "[Chinese Remainder Representation](https://en.wikipedia.org/wiki/Chinese_remainder_theorem)," converting inputs to their residues modulo many small primes. Then, we perform all modular products in parallel and then reconstruct the binary result via CRT. Working modulo the small primes turns the $n$-bit integers into $\log(n)$-bit residues, which fits the iterated adder we set up earlier. Proving that each step is in $\text{TC}^0$ is however quite involved. <d-cite key="hesse2001division"></d-cite>

### Matrix Multiplication

Matrix multiplication $AB = C$, $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n},$ can be decomposed into $mkn$ multiplications and then $mn$ iterated additions. Thus, the amount of computation is polynomial in the size of the input, and we can construct an algorithm for it in $TC_0$ by composing together the algorithms for iterated addition/multiplication. It should make sense that matrix multiplication is in $\text{TC}^0$ because is a highly parallel workload with almost no sequential dependency.

## Results for Different Architectures

### Transformers are in $\text{TC}^0$

The transformer is an embedding layer, a constant number of self-attention and MLP blocks, and an output layer. Each layer itself can be decomposed into a constant number of matrix multiplications / element-wise computations, which gives an intuitive sense for why transformers are in $TC_0$. Even though self-attention is an $O(T^2)$ matrix multiplication, we have shown that as long as the matmul is polynomial in $T$ it is in $TC_0$.

A computation that is not in $TC_0$ might involve not just increasing the *size* of the matrix multiplication as input size grows (as we do in attention), but rather layering on *more* layers of matrix multiplication as the input size grows. This is how RNNs and SSMs work, which is what we'll look at next.

### RNNs are $$\text{NC}^1$$-complete

Recurrent neural networks (RNNs) are defined by a recurrence of the form

$$
h_t=f(A h_{t-1}+Bx_t),\quad y_t=g(C h_t),
$$

for some nonlinear activation functions $f$ and $g$, and weights $A,B,C$. Crucially, the same operation is iterated $T$ times, with each iteration depending on the previous hidden state. This means the *depth* of the computation grows with the size of the input.

In fact, we can prove that RNNs are $\text{NC}^1$-complete, thus beyond $\text{TC}^0$ under the assumption that $\text{TC}^0\neq \text{NC}^1$ (recall that $C$-complete means that the computation is both in complexity class $C$ and $C$-hard, meaning at least as hard as every other computation in $C$). We know that RNNs are in $\text{NC}^1$ since they can be computed using the binary tree decomposition from earlier We can show this by showing that they can solve $S_5$ state tracking, which is known to be $\text{NC}^1$-hard.

{% include dd_parent_open.liquid title="Proof Sketch for $S_5$ being $\text{NC}^1$-hard" %}

We need to start with a problem that we know to be $\text{NC}^1$-hard, and then perform a reduction to $S_5$ state tracking.

In particular, we start with the *boolean formula evaluation problem*: 
> Given a formula $F(x_1,\ldots, x_n)$ of AND, OR, and NOT gates, is $F(x)=1$ for a given assignment?
Given any circuit in $\text{NC}^1$, we can convert it to a boolean formula evaluation problem by just duplicating the shared components of the circuit. Duplicating the shared parts just increases the size polynomially, but the depth stays $O(\log n)$.

Barrington, 1986. "permutation branching programs", a bit nontrivial but working on it -adam

{% include dd_parent_close.liquid %}

It is actually quite easy to show that RNNs can solve $S_5$ state tracking. Concretely, there are $120$ states that we need to track. We can use a $120$-dimensional hidden state $h_t$ in our RNN that represents each type of permutation as a one-hot basis vector. For each input permutation $a$, let $$P_a\in\{0,1\}^{120\times 120}$$ be the permutation matrix for applying $a$ through left-multiplication. We have $h_0=\text{identity permutation}$, and $h_t=P_{x_t}h_{t-1}$. 

We can construct an RNN with a weight matrix $A$ of dimensions $120^2\times 120$ that computes all the possible values of $P_ah_{t-1}$ in parallel. Then, we can have $B$ be a matrix of dimension $120^2\times 120$ that maps the one-hot input $x_t$ to a mask, with a $0$ in the $120$ entries that the input corresponds to and a large negative number in every other entry. Then, with a ReLU activation function $f$, the transition $h_t=f(A h_{t-1}+Bx_t)$ computes the permutation composition, storing the new state in $h_t$.

More generally, RNNs can simulate any deterministic finite automaton (DFA) in a similar way. The same reasoning extends to other RNN variants as well, such as LSTMs and GRUs. 

One of the big issues with these models though is that their sequential computation means an inability to have efficient parallelized training. Structured State Space Models (SSMs) came along as a parallelizable RNN.

### Some SSMs are in $$\text{TC}^0$$

A discrete-time generalized state space model with inputs $x_t$, hidden state $h_t$, and outputs $y_t$ can be written as

$$
h_{t+1} = A_t\, h_t + B_t\, x_t, \;\;
y_t = C_t\, h_t,
$$

where the matrices $A_t, B_t, C_t$ may be time-varying and/or depend on the input (e.g., $A_t = A(x_t)$). Note the lack of nonlinearity between layers.

We can also unroll the recurrence, which gives us the form

$$
h_t = \sum_{i=1}^t \left(\prod_{j=i+1}^t A_j\right) B_i x_i
$$

In the special case where $A_j = A$ (non time-varying), we can see that the above expression only uses iterated addition, multiplication, and *matrix-powering*, which is shown to also be in $\text{TC}^0$. <d-cite key="mereghetti2000threshold"></d-cite> This corresponds to the [S4](https://arxiv.org/pdf/2111.00396) model.

{% include dd_parent_open.liquid title="Proof Sketch for Matrix Powering" %}

The main insight is to use the Cayley-Hamilton theorem to reduce the problem of multiplying a matrix $n$ times to multiplying it only $k$ times for some constant $k$.

This theorem states that if we compute the characteristic polynomial for a matrix $M$: $$f(\lambda) = \det{(M - \lambda I)}$$

then $f(M) = 0$. Here, $f$ is an order-$k$ polynomial if $M \in \mathbb{R}^{k \times k}$.

Now, if we can factor $x^n = q(x) f(x) + r(x)$, then we know that $M^n = q(M) f(M) + r(M) = r(M)$. The remainder polynomial $r(M)$ is at most order $k-1$, so we have reduced the problem to computing a constant-size polynomial at $M$. The algorithm for performing polynomial arithmetic takes a similar spirit to the algorithms presented in the previous sections (e.g. polynomial multiplication is iterated addition and multiplication). <d-cite key="eberly1989veryfast"></d-cite>

{% include dd_parent_close.liquid %}

Additionally, when $A_j$ is diagonal or scalar times identity (which corresponds to [Mamba-2](https://arxiv.org/abs/2405.21060)), you are only using iterated addition and iterated multiplication, which implies that these special forms of the generalized SSM are also in $TC_0$.


### Some SSMs are not in $$\text{TC}^0$$

Delta-Net is not in $TC_0$. Instead of $A_t$ being a simple diagonal or constant matrix, it takes the form on identity plus low rank, and multiplying $n$ such matrices together is a non-trivial computation that no one has found an algorithm for in $TC_0$.