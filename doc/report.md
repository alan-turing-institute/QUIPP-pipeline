# QUIPP

## De-identification

### Re-identification

From page 6 of the [ICO Anonymisation Code of Practice](https://ico.org.uk/media/1061/anonymisation-code.pdf):

>We use the term ‘re-identification’ to describe the process of
turning anonymised data back into personal data through the use
of data matching or similar techniques.

### Personal data

From page 16 of the [ICO Anonymisation Code of Practice](https://ico.org.uk/media/1061/anonymisation-code.pdf): 
>The definition of ‘personal data’ can be difficult to apply in practice
for two main reasons:
>
>•	 the concept of ‘identify’ – and therefore of ‘anonymise’ - is not
straightforward because individuals can be identified in a number
of different ways. This can include direct identification, where
someone is explicitly identifiable from a single data source, such
as a list including full names, and indirect identification, where
two or more data sources need to be combined for identification
to take place; and
>
>•	 you may be satisfied that the data your organisation intends
to release does not, in itself, identify anyone. However, in
some cases you may not know whether other data is available
that means that re-identification by a third party is likely to
take place.
>
>In reality it can be difficult to determine whether data has been
anonymised or is still personal data. This can call for sensible
judgement based on the circumstances of the case in hand. This
code describes ways of assessing and mitigating the risks that
may arise, particularly in terms of assessing whether other data
is available that is likely to make re-identification likely. In some
cases, it will be relatively easy to determine whether it is likely
that a release of anonymised data will allow the identification
of an individual. In other cases it will be much harder, but the
decision still has to be made. 

### Anonymisation / Pseudonymisation

The boundary between these anonymisation and pseudonymisation is both fuzzy and drwn in different places by different organisations / references.

From page 7 of the [ICO Anonymisation Code of Practice](https://ico.org.uk/media/1061/anonymisation-code.pdf)
>We use the broad term ‘anonymisation’ to cover various techniques
that can be used to convert personal data into anonymised data.
We draw a distinction between anonymisation techniques used to
produce aggregated information, for example, and those – such
as pseudonymisation – that produce anonymised data but on an
individual-level basis. The latter can present a greater privacy risk,
but not necessarily an insurmountable one. We also draw a distinction
between publication to the world at large and the disclosure
on a more limited basis – for example to a particular research
establishment with conditions attached. 


From the [second Caldicott Review (2013)](https://www.gov.uk/government/publications/the-information-governance-review):
>**5.2 Anonymisation**
>
>Data ceases to be personal and confidential when it has been anonymised, as explained in 
chapter 6. In those circumstances, publication is lawful. Data, which has been anonymised
but still carries a significant risk of re-identification or de-anonymisation, may be treated
either as personal and confidential or as anonymised depending on how effectively the risk
of re-identification has been mitigated and what safeguards have been put in place. This is
fully explained in the ICO Anonymisation: Code of Practice and in chapter 6 of this report.

### Anonymisation / Pseudonimisation methods
From appendix 2 of the [ICO Anonymisation Code of Practice](https://ico.org.uk/media/1061/anonymisation-code.pdf)

>**Data masking**
>
>This involves stripping out obvious personal identifiers such
as names from a piece of information, to create a data set in
which no person identifiers are present. 

Note: Considered relatively high-risk by ICO because the
anonymised data still exists in an individual-level form. 

>**Pseudonymisation**
>
>De-identifying data so that a coded reference or pseudonym is
attached to a record to allow the data to be associated with a

Note: Considered relatively high-risk by ICO because the
anonymised data still exists in an individual-level form. 

>**Aggregation**
>
>Data is displayed as totals, so no data relating to or identifying
any individual is shown. Small numbers in totals are often
suppressed through ‘blurring’ or by being omitted altogether. 

Note: ICO includes Synthetic Data as a variant of Aggregation, but not of Pseudonymisation.

Note: Considered  low risk by ICO because it will
generally be difficult to find anything out about a particular
individual by using aggregated data. 

>**Derived data items and banding**
>
>Derived data is a set of values that reflect the character of the
source data, but which hide the exact original values. This is
usually done by using banding techniques to produce
coarser-grained descriptions of values than in the source
dataset eg replacing dates of birth by ages or years, addresses
by areas of residence or wards, using partial postcodes or
rounding exact figures so they appear in a normalised form.

Note: Considered  low risk by ICO because the
banding techniques make data-matching more difficult or
impossible.

### The 'motivated intruder' test

From pages 22-23 of the [ICO Anonymisation Code of Practice](https://ico.org.uk/media/1061/anonymisation-code.pdf)
>However a useful test – and one used by the Information Commissioner and the Tribunal that hears DPA and FOIA appeals –
involves considering whether an ‘intruder’ would be able to achieve
re-identification if motivated to attempt this.
>
>The ‘motivated intruder’ is taken to be a person who starts without any
prior knowledge but who wishes to identify the individual from whose
personal data the anonymised data has been derived. This test is
meant to assess whether the motivated intruder would be successful.
>
>The approach assumes that the ‘motivated intruder’ is reasonably
competent, has access to resources such as the internet, libraries,
and all public documents, and would employ investigative
techniques such as making enquiries of people who may have
additional knowledge of the identity of the data subject or advertising
for anyone with information to come forward. The ‘motivated
intruder’ is not assumed to have any specialist knowledge such as
computer hacking skills, or to have access to specialist equipment or
to resort to criminality such as burglary, to gain access to data that is
kept securely.

## Open publication vs limited access sharing

From page 37 of the [ICO Anonymisation Code of Practice](https://ico.org.uk/media/1061/anonymisation-code.pdf)
>It is important to draw a distinction between the publication of
anonymised data to the world at large and limited access. Clearly
the open data agenda relies on the public availability of data, and
information released in response to a freedom of information request
cannot be restricted to a particular person or group. However, much
research, systems testing and planning, for example, takes place by
releasing data within a closed community, ie where a finite number
of researchers or institutions have access to the data and where its
further disclosure is prohibited, eg by a contract. The advantage of
this is that re-identification and other risks are more controllable, and
potentially more data can be disclosed without having to deal with
the problems that publication can cause. It is therefore important to
draw a clear distinction between:
>
>•	 publication to the world at large, eg under the Freedom of
Information Act 2000 or open data. Here – in reality - there is
no restriction on the further disclosure or use of the data and no
guarantee that it will be kept secure; and
>
>•	 limited access, eg within a closed community of researchers. Here
it is possible to restrict the further disclosure or use of the data
and its security can be guaranteed.
>
>Limited access is particularly appropriate for the handling of
anonymised data derived from sensitive source material or where
there is a significant risk of re-identification.
>
>There can still be risks associated with limited access disclosure -
but these can be mitigated where data is disclosed within a closed
community working to established rules. Data minimisation rules will
also remain relevant.
>
>It could be appropriate that data anonymised from a collection of
personal data is published, whilst a record-level version of the data is
released in a limited way under an end-user agreement. 

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



## simPop notes

#### Summary

simPop is an R package that combines a number of methods to perform static spatial microsimulation. Following the typical microsimulation scenario, it generates synthetic populations by combining individual-level samples (i.e. microdata or seed) and aggregated census data (i.e. macrodata or target). Its intended use is for cases where spatial microsimulation is desirable, i.e. where the individuals belong to household and regional hierarchical structures. Nevertheless, the functions available can be used for more general synthetic tasks without a need for this hierarchy to be present (although some of the function arguments related to households still need to be filled). In terms of methodology, the package provides implementations of IPF, Simulated Annealing and model-based data  synthesis methods for categorical, semi-continuous and continuous variables (both parametric and non-parametric). It also offers some other methods which are useful for creating household structures and allows the user to tune various parameters (e.g. the sequence of variable synthesis). It comes with its own example data set which consists of a sample from an Austrian census and some aggregated data.


#### Methodology

The package performs spatial microsimulation. The purpose of spatial microsimulation is to synthesise individual-level data allocated to geographical regions, based on constraints. The synthesised data need to be as big as the whole population (or the part of the population that we are interested in). Usually, we only have access to the following to sets of data from which we want to synthesise:
-	Micro-data: These are individual-level data which are a sample from the population, e.g. 1% of all census data. They contain information for each individual, e.g. gender, age, region, income. This data set is also called the ‘seed’ in microsimulation literature. We are trying to synthesise a population-size extension of this data set, allocated to regions based on constraints. These constraints are provided by the macro-data.
-	Macro-data: These are spatially (or otherwise) aggregated data from the same source (e.g. census) which provide the constraints based on which we are going to synthesise. They are cross-tabulations, e.g. frequencies of individuals who are male/female and live in each of the 9 regions of England. These are called the ‘target’ in literature. The synthesised population needs to adhere to the constraints posed by the target, e.g. the number of males in Greater London in the synthesised population needs to be the same or similar to what we have in the macro-data.

Note that simPop assumes that each individual micro-datum in the imported set has a weight (initial weight). The weights can be interpreted as “an individual with weight w is going to be replicated w times when forming the synthesised population”. It also assumes the micro-data have a household structure and stratum (region).

The main simPop flow to create a synthetic population data set is the following:
-	Data Import: Import the individual-level micro-data using specifyInput() and the aggregated macro-data as a table.
-	Sample calibration: Calibrate the micro-data using calibSample(). Calibration consists in editing the weights of the individuals in the micro-data so as to reflect the constraints in the macro-data. This kind of calibration is done using the IPF reweighting algorithm – details can be found in several microsimulation papers in Zotero.
-	Population extension: Extend the data set to population size by resampling the household structure and some basic variables using simStructure(). These basic variables should be as few as possible to avoid compromising privacy and they should be the ones that are the least sensitive to intruder attacks. simStructure() creates realistic household structures since it resamples the structures it sees in the micro-data.
*	Variables synthesis: Synthesize the remaining variables in the data set (categorical and continuous) using simCategorical() and simContinuous(). These are allowed to have non-observed relationships to households, as the sample is unlikely to capture all possible (reasonable) relationships in the population. The package provides implementations of parametric and non-parametric models for categorical, semi-continuous and continuous variables (e.g. multinomial, logistic and two-step regression, tree-based methods). The synthesis proceeds as follows for each synthesised variable: 
    * A model is trained to predict the synthesised variable given all variables that have been synthesised so far. Training is done on the sample data set so all data are “original - real” data at this stage.
    *	The variable is synthesised by scoring the trained model. The synthetic data are used in this stage as input to the trained model (i.e. the data that have already been synthesised in previous steps).
    * The process is repeated for all variables.
The order of variables’ data synthesis can be modified. Users can apply corrections to age variables (Whipple, Sprague indices; corrects for over-reporting of ages ending in 5 or 0). Multiple methods for generating categorical variables are available: multinomial logistic regression, random draws from the conditional distribution or classification trees. Likewise, multiple methods are available for continuous data: 1) Fitting a multinomial model and taking random draws from the intervals of the categories into which predictions fall, 2) Apply a logistic regression to separate close-to-zero values from the rest. Then for non-zeros  apply a linear regression. Random noise is also added - either based on normal assumption or by sampling from the residuals. 
-	Other synthesis tasks: Simulate variables’ components and/or allocate the population to small areas with simComponents() and simInitSpatial(). 
    *	The first one is used to simulate components that together add up to one variable, e.g. different sources of income that are combined to make up the total income. The inputs are the sample data containing all the components and the synthetic (population data) where only the combined variable is available. The output are the components in the population data. 
    *	The second one is used to simulate granular geographical information if needed. Using simInitSpatial we can simulate information on a finer level than the one we already have - e.g. districts instead of regions. This requires a table that contains the known population of smaller areas for each larger area, either as number of persons or number of households.
-	Population calibration: Calibrate the (now fully synthesised) population data set using calibPop() if required. This uses Simulated Annealing (SA) and requires some form of cross-tabulated census data (constraints) to calibrate against. SA is an iterative optimisation algorithm which swap households between regions in each step to converge to a solution that closely adheres to the constraints. 

The above flow can be altered depending on requirements and some of the functions can be used independently but with some care. For example:
-	specifyInput + calibSample: Can be used to simply run IPF on any data set, regardless of whether the data set is a sample or a population. Initial weights need to be provided, as well as a household id variable. 
-	specifyInput + simStructure + simCategorical + simContinuous: Can be used to simply synthesise data with model-based  synthesis, without any calibration (IPF, etc). The size of the synthesised data set depends on the weights the initial data set  has been assigned (simStructure "replicates" individuals a number of times equal to their weights). If weights are all equal to 1.0 the synthesised data set will have the same size as the original. Note that synthesis is done once and not multiple times like in multiple imputation algorithms.
 
The package has a number of settings and features, e.g. allows parallelisation for several functions and various changes to model training parameters.

#### Utility and privacy
The package offers several utility metrics to compare the original micro-data sample with the synthesised micro-data population. The basis for comparison are the frequencies (counts) for various subsets (groupings) of data, e.g. count of women in Greater London. For the sample data, a weighted mean is computed (using the weights) instead of the simple counts in order to get estimates of the expected population counts.

The utility metrics include Mosaic, CDF and Box plots and a comparison of the results of a regression trained on the original and synthesised data set (checks if estimated coefficient CIs overlap).

#### Pros and cons
Pros:
-	simPop handles complex data structures such as individuals within households when doing microsimulation
-	Has a lot of features and settings and implements many methods (microsimulation techniques, parametric and non-parametric models)
-	Processing speed is fast due to ability to use parallel CPU processing
-	Marginal distributions can be well preserved as they are used in the synthesising process
-	Offers utility metric functions, some of which are visualisations.
-	Documentation is decent and the code seems well written

Cons:
-	Flow is tailored for the specific case of simulating households and is a bit inflexible although some components are reusable.
-	No multiple imputation option when doing model-based synthesis.
-	Lack of privacy-related features/metrics/tuning


