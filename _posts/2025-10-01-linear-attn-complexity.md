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

bibliography: complexity-theory.bib


---

A lot of recent models such as [DeltaNet](https://sustcsonglin.github.io/assets/pdf/talk_250117.pdf) and [RWKV-7](https://arxiv.org/abs/2503.14456) make claims about moving beyond the $$\text{TC}^0$$ complexity class, which the Transformer is confined to. This blog post will dig into what the claims actually mean, and why they may/may not be relevant.

## Motivation: State Tracking

When we analyze the computational power of different architectures, it helps to ground the discussion in concrete problems we care about. One very important class of problems is *state tracking*: given a sequence of inputs that update some underlying system, can the model keep track of the current state of that system?

A canonical example of this is group composition, in particular on the symmetric group on 5 elements, $S_5$. The state tracking problem for $S_5$ is: given a sequence of permutations on 5 elements, compute the final state of the elements (the composition of the permutations). For example: 
> Start with the ordering 1,2,3,4,5. Apply the following permutations: swap elements 1 and 3; swap elements 5 and 3; rotate elements 1, 3, and 4.

The final state is then $4, 2, 3, 5, 1$. This type of problem is often referred to as the word problem for a group (to be precise, it is: given a sequence of group elements, is its product the identity?)

This type of problem shows up everywhere. For example, imagine trying to track the state of a chessboard as a sequence of moves is played, or the state of a codebase as a sequence of diffs is applied.

From a complexity-theoretic perspective, these problems are quite significant. The state-tracking problem for $S_5$ is known to be $\text{NC}^1$-complete, meaning it is "at least as hard" as any other problem in the complexity class $\text{NC}^1$. By contrast, problems in the weaker class $\text{TC}^0$ cannot simulate this task. As it turns out, an example of a problem that is in the weaker $\text{TC}^0$ class is *computing the result of a transformer's forward pass*. Hence, it can be shown that transformers cannot solve state-tracking. 

The goal of this blog post is to formally prove this statement, give some intuition as to why certain architectures can and cannot track state, and analyze any implications of this for designing expressive architectures.


## A Background on Circuit Complexity Classes

Our computation model is that we have a set of gates wired together that take in boolean inputs and compute some function of the inputs. Below is an example circuit with $$5$$ gates, $$3$$ inputs, and a depth of $$3$$.

{% include figure.liquid loading="eager" path="assets/img/1.webp" class="img-fluid rounded z-depth-1" %}

Circuits are in $$\text{NC}^k$$ if for $$n$$ input variables, they have depth at most $$O((\log{n})^k)$$. Here, we are allowed to use any 2-input or 1-input gate, such as AND, OR, XOR, NOR, etc. These are gates with bounded "fan-in" (number of inputs).

One example of a problem in $\text{NC}^1$ we gave earlier is state-tracking on $S_5$. Another example of a common problem that is in $$\text{NC}^1$$ is computing the parity of $$n$$ bits. We can compute the parity of two bits through $x_i \oplus x_j$, and we can extend this to arbitrary sized inputs by constructing a binary tree:

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

<!-- add some space here somehow -->

A circuit is in $$\text{TC}^0$$ if it has **constant depth** and a polynomial number of gates. Constant depth is severely limiting, however, so we are additionally allowed to use a *threshold* gate in addition to the usual 2-input boolean ones. A threshold gate is defined as $$G(x_1, x_2, ..., x_n) = \textbf{1}_{\sum_i w_i x_i \geq \theta}$$ where the weights $$\textbf{w}$$ and threshold $$\theta$$ are parameters of the gate (rather than inputs). This is very powerful. For example, this gives us the ability to construct an AND with unbounded fan-in in a single gate by setting $$w_i = 1$$ and $$\theta = n$$. Thus, many of the circuits that would have required a logarithmic-depth binary tree can now be constructed in constant depth using threshold gates.

Despite the additional power that threshold gates provide, it is proven that $$\text{TC}^0 \subseteq \text{NC}^1$$. Whether this containment is strict or if it is an equality is still an open question. It is often assumed that the containment is strict, though Ryan Williams actually puts it at 50/50. <d-cite key="williams2022estimatedlikelihoods"></d-cite>

Finally, many papers also refer to $$\textit{L}$$-*uniform* circuit families, which are circuits that can be generated by a deterministic Turing machine running in logarithmic space. This condition is often added to rule out hardwired circuit designs.

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

It's not immediately obvious how to add two numbers in constant depth since the algorithm we are taught in elementary school is inherently sequential: go right-to-left, adding each digit and computing a *carry* for the next column of digits. This is an $O(n)$ sequential algorithm for $n$-bit numbers. However, we *can* add two numbers in constant depth using [carry-lookahead addition](https://en.wikipedia.org/wiki/Carry-lookahead_adder). To add two numbers $a_n a_{n-1} .. a_{0}$ and $b_n b_{n-1} .. b_{0}$, we compute the propagator bits $p_i$ and generator bits $g_i$. The generator bits indicate that the next most significant bit will have a carry of one *regardless* of the carry of the current bit, and the propagator bits indicate that the next bit will have a carry of $1$ if the current bit has a carry of $1$.

$$
p_i = a_i \lor b_i, \; \; g_i = a_i \land b_i
$$

$$
c_i = \textbf{1}_{\exists j \mid p_{i-1} \land p_{i-2}... \land p_{j+1} \land g_j}
$$

Thus computing the carry bit requires computing the propagators and generators, along with expressions of the form $p_{i-1} \land p_{i-2}... \land p_{j+1} \land g_j$. This is an AND with many inputs, which we've already shown can be computed with a single threshold gate! We can also compute an OR with unbounded fan-in by setting $\textbf{w} = 1$ and $\theta=1$, which we will use to check if at least one AND expression is satisfied. So now we can add two numbers in $TC_{0}$!

### Iterated Addition

Now, let's look at the problem of adding $n$ numbers which are each $n$ bits.

We can start with the simpler problem of adding $n$ bits ($b_1, ... b_n$). First observe that we can construct a threshold for each of the $n$ possible values of the sum:
$$
\mathbf{1}_{\sum_j b_j \geq i}
$$. Then, we can construct a one-hot vector $\bf{o}$ for the value of the sum: $$o_i = \textbf{1}_{\sum_j b_j \geq i} \land \textbf{1}_{\sum_j -b_j \geq i}$$. Then, this $n$-dimensional one-hot vector can be mapped to the exact output in a single layer of threshold circuits.

In fact, this approach works whenever the number of possible values of the sum is polynomial in $n$ (since we are only allowed a polynomial number of gates). Thus, we can use the exact same approach for any polynomial number of $O(\log{n})$-bit numbers (for the threshold sums, we can weight a bit in position $i$ with weight $2^i$).

The last step is to extend this to $n$-bit numbers. Consider the $n \times n$ bit array that stacks all of the numbers on top of each other. We can organize these bits into groups of $\log{n}$ columns, and label each group as even or odd in a striped fashion. When we add a single even group, the result will have at most $2 \log{n}$ bits, which means that the it would not reach the next even group. Thus, we can add all of the bits in all of the even groups, and then add the odd groups, and issue a final addition to add then together.


Multiplication can be reformulated as iterated addition, which implies that multiplying two numbers is also in $TC_0$.


### Iterated Multiplication

The algorithm for iterated multiplication is slight more complicated, but it is also in $TC_0$. The proof is in Allender and Ogihara. <d-cite key="allender1999division"></d-cite>


## Some SSMs are in $$\text{TC}^0$$

A discrete-time generalized state space model with inputs $x_t$, hidden state $h_t$, and outputs $y_t$ can be written as

$$
h_{t+1} = A_t\, h_t + B_t\, x_t, \;\;
y_t = C_t\, h_t,
$$

where the matrices $A_t, B_t, C_t$ may be time-varying and/or depend on the input (e.g., $A_t = A(x_t)$).

We can also unroll the recurrence, which gives us the form

$$
h_t = \sum_{i=1}^t \left(\prod_{j=i+1}^t A_j\right) B_i x_i
$$

In the special case where $A_j = A$ (non time-varying), we can see that the above expression only uses iterated addition, multiplication, and *matrix-powering*, which is shown to also be in $\text{TC}^0$. <d-cite key="mereghetti2000threshold"></d-cite> This corresponds to the [S4](https://arxiv.org/pdf/2111.00396) model.

Additionally, when $A_j$ is diagonal or scalar times identity (which corresponds to [Mamba-2](https://arxiv.org/abs/2405.21060)), you are only using iterated addition and iterated multiplication, which implies that these special forms of the generalized SSM are also in $TC_0$.


## Some SSMs are not in $$\text{TC}^0$$

Delta-Net is not in $TC_0$. Instead of $A_t$ being a simple diagonal or constant matrix, it takes the form on identity plus low rank, and multiplying $n$ such matrices together is a non-trivial computation that no one has found an algorithm for in $TC_0$.