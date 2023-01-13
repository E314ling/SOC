# SOC

This project tests different reinforcement learning algorithms for stochastic optimal control problems. Namely they are used to solve a stochastic linear quadratic regulator problem in two different szenarios.
## Szenario one: 
We just solve

$$
X_{n+1} = AX_n+ B u_n+  \sigma \xi_{n+1}
$$

where $X_n$ is the state at time $n$, $u_n$ is the control at time $n$ and $\xi_{n+1} \sim \mathcal{N}(0,1)$. We solve this problem for a time horizon $N \in \mathbb{N}$ and identify $n = t_n$
with $t_n = n \cdot  \Delta t$. This  equation has a closed-form solution which we use to compare the results generated by the algorithms.
We aim to minimize

$$
J(x) = \mathbb{E} \bigg( \sum_{i=0}^{N-1} \Delta t (X_i^T Q X_i + u_i^T R u_i ) + X_N^T D X_N \big | X_0 = x \bigg)
$$
## Szenario two:

Here we try to solve the continous SOC

$$
d X_t = (A X_t + Bu_t)dt + \sigma dB_t
$$

with $t \in [0, \tau]$ where $\tau := \min \lbrace{ N, \underset{t \geq 0}{\inf} \lbrace{ t: X_t \notin S \rbrace} \rbrace}$,  $S = \lbrace{ X \in \mathbb{R}^n | \parallel X \parallel \in [1,3] \rbrace}$.
Discretizing with Euler-Maruyama get us the time discrete problem

$$
X_{n+1} =  X_n + (AX_n+ B u_n) \Delta t +  \sigma \sqrt{\Delta t} \xi_{n+1}
$$

We aim to minimize

$$
J(x) = \mathbb{E} \bigg( \sum_{i=0}^{\tau-1} (X_i^T Q X_i + u_i^T R u_i ) + X_\tau^T D X_\tau \big | X_0 = x \bigg)
$$
