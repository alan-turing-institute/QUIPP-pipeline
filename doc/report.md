# QUIPP

## Assessing privacy

### Differential privacy

A function $A$, from a dataset in $\mathcal{D}$, to a random variable taking values in $R$, is said to be
\emph{$\epsilon$-differentially private} if for any $S \subseteq R$, and where $D'$ is any dataset derived from $D \in
\mathcal{D}$ by the addition or removal of a single element, the following holds:

$$\mathbf{P}(A(D) \in S) \le e^{\epsilon}\mathbf{P}(A(D') \in S)$$ {#eq:diff-priv-eps}

A function is said to be \emph{$(\epsilon,\delta)$-differentially private} if

$$\mathbf{P}(A(D) \in S) \le e^{\epsilon}\mathbf{P}(A(D') \in S) + \delta$$ {#eq:diff-priv-eps-delta}

When probability density functions exist for these distributions, this is equivalent to

$$p(x) \le e^{\epsilon} q(x)\; [+ \delta]$$

for all $x \in S$, where
$$
p(x) = f_{A(D)}(x) \\
q(x) = f_{A(D')}(x).
$$

An alternative way of writing the condition @eq:diff-priv-eps is
$$
\log \left\( \frac{p(x)}{q(x)} \right\) \le \epsilon,
$$
and @eq:diff-priv-eps-delta follows from the condition
$$
p(x) \le \delta or \log \left\( \frac{p(x)}{q(x)} \right\) \le \epsilon.
$$

In the context of data synthesis, $A$ takes the (private) data as input and produces a distribution, with synthetic data
produced by sampling from this distribution.

In principle, for a given dataset, it is possible to check the above property empirically, by comparing the distribution of
$A(D')$ with that of $A(D)$, although this may be impractical.

#### Example




## Utility metrics

The general scenario is that we have two data sets and we want to assess whether they are equally useful for our analysis or purposes:
- The original data set, which we do not want to release due to privacy or other concerns. This could come from a database or a survey. In the case of microsimulation, the original data could also be a sample from a population (e.g. in the case of a census), where we have access only to the sample and not the population.
- The synthesised data set, which is created using any of a number of synthesis methods and contain "fake" data which we want to be able to release. In the case of microsimulation, the synthesised data set is usually an approximation of the population that we do not have and the aim is to use this to do our analysis because the sample is too small.

Note that in the case of microsimulation (where only a sample from the real population is observed/known), it might be appropriate to apply bootstrapping or other similar methods to the sample in order to grow its size to match the size of the synthetic population. We then use this "scaled up" sample when comparing with the synthetic population using the metrics given below.

There are many utility metrics to compare the usefulness of the two data sets and some of them are application-specific. This is a list of metrics that could be applied to the problem, some of which have been taken from literature. 

#### Descriptive statistics: 
Various descriptive statistics for each variable in the data can be evaluated on the original/synthesised data sets and then compared by finding relative differences or performing hypothesis tests (e.g. equality of means), etc. Examples: Mean, variance, moments, modes, percentiles, min/max values.

#### Correlations
Correlation tables can show if the correlation structure of the data set is maintained between all variables. For categorical variables, various methods based on contingency tables are recommended by different authors (e.g. phi coefficient). Again, comparing between the two sets could be done by finding relative differences or some other error metric or by performing statistical tests (e.g. Box M or Steiger's method implemented in R package cortest).

#### Goodness of fit tests and metrics
- The empirical CDFs of each variable on the original and synthesised data set can be compared using the Kolmogorov-Smirnoff test. It is a simple test that uses the maximum vertical distance between the CDFs to compute its statistic. The Cramer-von-Mises and Kuiper tests are alternatives.
- For PDFs it is possible to use the Kullback-Leibler divergence. It can be applied to the joint distribution of all variables in the data or or to each variable separately. The distance goes to zero when the PDFs are the same. Unfortunately it is compuatationally intensive to compute. Under normality assumption it has a closed form. Without the normality assumption it can be evaluated using a combination of Kernel Desnity Esitmation and numerical integration or MCMC sampling from the joint distributions. The advantage is that it captures "all" the infoormation of the data set and not just part of it like a desriptive statistic or a correlation do.
- Other statistical tests include the Shapiro-Wilk (testing for normality assuming we want that), Anderson-Darling (tests if the sample belongs to a given distribution), Pearson's chi-squared test (for categorical data).

#### Regression for goodness of fit
This involves fitting a regression model on the original data set (with a selected variable as response) and then using the data in the synthesised data set to predict values for the same variable. Various accuracy metrics can be evaluated to measure goodness of fit (e.g. check the residuals for constant mean and variance, all the typical regression statistics). This is effectively the approach followed by many synthsis packages tto synthesise data but it can be used for utility evaluation too. This can be done with various models including ML models but with less theoretical stats available.

#### Frequencies
This involves finding the counts for different combinations of the variables, e.g. number of individuals that are male, adult and live in London and comparing these counts between the two data sets. When one of the data sets has smaller size (e.g. in microsimulation), it is recommended to oversample or use the weights of the individuals.

#### Outliers
Find outliers using standard methods and compare if their frequencies and values are the same in both sets.

#### Missing values
Check if the frequencies of missing values are the same. Check if the pattern of appearance of missing value is the same (e.g. in the same combinations of other variables).

#### Visualisations
Visually inspect similarity in graphs:
- Frequency plots
- Mosaic plots
- Empirical CDFs
- Boxplots
- Heat maps with correlations
- Outlier plots
- QQ plots
- ...

#### Regression/Classification model comparison
This involves training the same model on the original and synthesised data and comparing the results. We expect to see similar outcomes if the data sets are similar. Many models can be trained with different variabels involved (including interactions, etc). To compare, we can check estimated coefficients and their confidence intervals and how they overlap (for GLMs), switches between positive and negative values in the same coefficient (GLMs), switches between significant and non-significant (GLMs), residual distributions (any model), feature imporances (tree based methods), weights (neural networks - unrealistic?). For CI overlap, there are various metrics proposed by the authors of synthpop. We can also do different forms of cross validation a nd compare the results.

#### Holdout based comparison
We can hold out a chunk of the original data set and then use the redacted original to synthesise. We then train the same model on the redacted original and the synthesised and predict the holdout. We can compare the predictions, residuals, feature importance (for tree-based methods) and other metrics to see if they are the same. For classification problems, we can check whether the same proportion of observations fall into the same class for original/synthetic.

#### Clustering
The datasets should behave the same way for a variety of uses/analyses (although there might be users that are only interested in specific analyses). Unsupervised tasks could be used for comparison, e.g. do the two data sets lead to approximately the same clusters being found when k-means is applied? One approach to compare datasets using clustering is given in https://doi.org/10.29012/jpc.v1i1.568

#### Dimensionality reduction
Does PCA (and other more complex methods) give the same principal components, etc?

#### Non-parameteric models
Try K-Nearest Neighbours for regression/classification and see if the results are similar.

#### Optimisation
Solve simple optimisation problems which use the variables of the data set and see if the solutions are the same, e.g. minimise some formula with gradient descent, solve a toy MILP problem (could be tricky?)

#### Variable selection behaviour
Does LASSO drop the same variables when trained on the two data sets?

#### Propensity Score
As described in https://doi.org/10.29012/jpc.v1i1.568, this is a score that uses logistic regression predicting membership to the original/synthetic sets given all the variables in the datasets. The resulting probabilities are compared based on their percentiles.

#### Other methods
Many other ML methods can be tested, e.g. sklearn has a bunch of them under the same API. 

#### Weighted average of many metrics
Almost all of the above metrics are quantifiable and thus can be combined into a weighted "total score" which would give the user a feel about overall utility. The user could choose which metrics they want to include in the score and how important each metric is.



## Why traditional privacy preservation methods are insufficient
(taken from https://doi.org/10.1016/j.csda.2011.06.006)
To protect confidentiality in public use datasets, many statistical agencies release data that have been altered to protect confidentiality. Common strategies include aggregating geography, top-coding variables, swapping data values across records, and adding random noise to values. As the threats to confidentiality grow, these techniques may have to be applied with high intensity to ensure adequate protection. However, applying these methods with high intensity can have serious consequences for secondary statistical analysis. For example, aggregation of geography to high levels disables small area estimation and hides spatial variation; top-coding eliminates learning about tails of distributions—which are often most interesting—and degrades analyses reliant on entire distributions; swapping at high rates destroys correlations among swapped and not swapped  ariables; and, adding random noise introduces measurement error that distorts distributions and attenuates correlations. In fact, Elliott and Purdam (2007) use the public use files from the UK census to show empirically that the quality of statistical analyses can be degraded even when using recoding, swapping, or stochastic perturbation at modest intensity levels. These problems would only get worse with high intensity applications.

Also, a utility and privacy comparison between traditional methods can be found in https://doi.org/10.1198/000313006X124640
