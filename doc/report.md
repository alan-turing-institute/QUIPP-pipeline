# QUIPP

## Assessing privacy

### Differential privacy

A function $A$, from a dataset in $\mathcal{D}$, to a random variable taking values in $R$, is said to be
\emph{$\epsilon$-differentially private} if for any $S \subseteq R$, and where $D'$ is any dataset derived from $D \in
\mathcal{D}$ by the removal of a single element, the following holds:

$$\mathbf{P}(A(D) \in S) \le \e^{\epsilon}\mathbf{P}(A(D') \in S)$$

In the context of data synthesis, $A$ takes the (private) data as input and produces a distribution, where samples
from this distribution corresponding to (disclosive) synthetic data.


