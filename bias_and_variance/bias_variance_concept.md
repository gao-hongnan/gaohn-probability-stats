# Bias-Variance Tradeoff Concept

## Learning Objectives
- Understand the concepts of bias, variance, and irreducible error.
- Learn how the bias and variance of a model relate to the complexity of a model.
- Visualize the tradeoff between bias and variance.
- Why does increasing number of sample decrease the Variance in the Bias-Variance tradeoff, and therefore, an effective way to curb overfitting?

## Motivation

> **Learning from Data**

> Think of an ideal situation where your hypothesis set $\mathcal{H}$ has a singleton hypothesis $h$ where $h = f$, the ground truth function. In this way, we can safely say both the bias and variance of $h$ is 0 as there is no error to begin with (exclude irreducible error). However, this situation is not going to happen in the real world. As a result, we resort to slowly increasing the size of $\mathcal{H}$ (i.e. using a larger model with more degrees of freedom), in an attempt to hope that as $|\mathcal{H}|$ increases, the chance of our target function sitting in $\mathcal{H}$ increases too.

---

> Therefore, we need to strike a balance between achieving the below two points:
>> 1. To have some hypothesis set $\mathcal{H}$ such that there exists $h \in \mathcal{H}$ that approximates $f$ as well as possible. In this context, as well as possible refers to the mean squared error.
>> 2. Enable the data to zoom in on the right hypothesis.


## Overview of Bias-Variance Errors

### The Regression Setup

Consider the general regression setup where we are given a random pair
$(X, Y) \in \mathbb{R}^p \times \mathbb{R}$. We would like to
\"predict\" $Y$ with some **true** function of $X$, say, $f(X)$.

To clarify what we mean by \"predict,\" we specify that we would like
$f(X)$ to be \"close\" to $Y$. To further clarify what we mean by
\"close,\" we define the **squared error loss** of estimating $Y$ using
$f(X)$.

$$
L(Y, f(X)) \triangleq (Y - f(X)) ^ 2
$$

Now we can clarify the goal of regression, which is to minimize the
above loss, on average. We call this the **risk** of estimating $Y$
using $f(X)$.

$$
R(Y, f(X)) \triangleq \mathbb{E}[L(Y, f(X))] = \mathbb{E}_{X, Y}[(Y - f(X)) ^ 2]
$$

The above is our favourite Mean Squared Error Loss, where we sum up all
the squared error loss of each prediction and its ground truth target,
and take the average of them.

Before attempting to minimize the risk, we first re-write the risk after
conditioning on $X$.

$$
\mathbb{E}_{X, Y} \left[ (Y - f(X)) ^ 2 \right] = \mathbb{E}_{X} \mathbb{E}_{Y \mid X} \left[ ( Y - f(X) ) ^ 2 \mid X = x \right]
$$

Minimizing the right-hand side is much easier, as it simply amounts to
minimizing the inner expectation with respect to $Y \mid X$, essentially
minimizing the risk pointwise, for each $x$.

It turns out, that the risk is minimized by setting $f(x)$ to be equal
the conditional mean of $Y$ given $X$,

$$
f(x) = \mathbb{E}(Y \mid X = x)
$$

which we call the **regression function**.\^\[Note that in this chapter,
we will refer to $f(x)$ as the regression function instead of $\mu(x)$
for unimportant and arbitrary reasons.\]

Note that the choice of squared error loss is somewhat arbitrary.
Suppose instead we chose absolute error loss.

$$
L(Y, f(X)) \triangleq | Y - f(X) | 
$$

The risk would then be minimized setting $f(x)$ equal to the conditional
median.

$$
f(x) = \text{median}(Y \mid X = x)
$$

Despite this possibility, our preference will still be for squared error
loss. The reasons for this are numerous, including: historical, ease of
optimization, and protecting against large deviations.

Now, given data
$\mathcal{D} = (x_i, y_i) \in \mathbb{R}^p \times \mathbb{R}$, our goal
becomes finding some $\hat{f}$ that is a good estimate of the regression
function $f$. We\'ll see that this amounts to minimizing what we call
the reducible error.

### Expected Generalization/Test Error

> **Author here uses $\hat{f}$ to represent hypothesis $h$.**

------------------------------------------------------------------------

Suppose that we obtain some $\hat{f}$, how well does it estimate $f$? We
define the **expected prediction error** of predicting $Y$ using
$\hat{f}(X)$. A good $\hat{f}$ will have a low expected prediction
error.

$$
\text{EPE}\left(Y, \hat{f}(X)\right) \triangleq \mathbb{E}_{X, Y, \mathcal{D}} \left[  \left( Y - \hat{f}(X) \right)^2 \right]
$$

This expectation is over $X$, $Y$, and also $\mathcal{D}$. The estimate
$\hat{f}$ is actually random depending on the data, $\mathcal{D}$, used
to estimate $\hat{f}$. We could actually write $\hat{f}(X, \mathcal{D})$
to make this dependence explicit, but our notation will become
cumbersome enough as it is.

Like before, we\'ll condition on $X$. This results in the expected
prediction error of predicting $Y$ using $\hat{f}(X)$ when $X = x$.

$$
\text{EPE}\left(Y, \hat{f}(x)\right) = 
\mathbb{E}_{Y \mid X, \mathcal{D}} \left[  \left(Y - \hat{f}(X) \right)^2 \mid X = x \right] = 
\underbrace{\mathbb{E}_{\mathcal{D}} \left[  \left(f(x) - \hat{f}(x) \right)^2 \right]}_\textrm{reducible error} + 
\underbrace{\mathbb{V}_{Y \mid X} \left[ Y \mid X = x \right]}_\textrm{irreducible error}
$$

A number of things to note here:

-   The expected prediction error is for a random $Y$ given a fixed $x$
    and a random $\hat{f}$. As such, the expectation is over $Y \mid X$
    and $\mathcal{D}$. Our estimated function $\hat{f}$ is random
    depending on the data, $\mathcal{D}$, which is used to perform the
    estimation.
-   The expected prediction error of predicting $Y$ using $\hat{f}(X)$
    when $X = x$ has been decomposed into two errors:
    -   The **reducible error**, which is the expected squared error
        loss of estimation $f(x)$ using $\hat{f}(x)$ at a fixed point
        $x$. The only thing that is random here is $\mathcal{D}$, the
        data used to obtain $\hat{f}$. (Both $f$ and $x$ are fixed.)
        We\'ll often call this reducible error the **mean squared
        error** of estimating $f(x)$ using $\hat{f}$ at a fixed point
        $x$. $$
        \text{MSE}\left(f(x), \hat{f}(x)\right) \triangleq 
        \mathbb{E}_{\mathcal{D}} \left[  \left(f(x) - \hat{f}(x) \right)^2 \right]$$
    -   The **irreducible error**. This is simply the variance of $Y$
        given that $X = x$, essentially noise that we do not want to
        learn. This is also called the **Bayes error**.

As the name suggests, the reducible error is the error that we have some
control over. But how do we control this error?

### Reducible and Irreducible Error

As mentioned in the previous section, our **Expected Test Error** in a
Regression Setting is given formally as follows:

More formally, in a regression setting where we Mean Squared Error,
$$\begin{aligned}\mathcal{E}_{\text{out}}(h) = \mathbb{E}_{\mathrm{x}}\left[(h_{\mathcal{D}}(\mathrm{x}) - f(\mathrm{x}))^2 \right]
\end{aligned}$$

------------------------------------------------------------------------

This is difficult and confusing to understand. To water down the formal
definition, it is worth taking an example, in
$\mathcal{E}_{\text{out}}(h)$ we are only talking about the **Expected
Test Error** over the Test Set and nothing else. **Think of a test set
with only one query point**, we call it $\mathrm{x}_{q}$, then the above
equation is just
$$\begin{aligned}\mathcal{E}_{\text{out}}(h) = \mathbb{E}_{\mathrm{x}_{q}}\left[(h_{\mathcal{D}}(\mathrm{x}_{q}) - f(\mathrm{x}_{q}))^2 \right]
\end{aligned}$$

over a single point over the distribution $\mathrm{x}_{q}$. That is if
$\mathrm{x}_{q} = 3$ and $h_{\mathcal{D}}(\mathrm{x}_{q}) = 2$ and
$f(\mathrm{x}_{q}) = 5$, then
$(h_{\mathcal{D}}(\mathrm{x}_{q}) - f(\mathrm{x}_{q}))^2 = 9$ and it
follows that
$$\mathcal{E}_{\text{out}}(h) =  \mathbb{E}_{\mathrm{x}_{q}}\left[(h_{\mathcal{D}}(\mathrm{x}_{q}) - f(\mathrm{x}_{q}))^2 \right] = \mathbb{E}_{\mathrm{x}_{q}}[9] = \frac{9}{1} = 9$$

Note that I purposely denoted the denominator to be 1 because we have
only 1 test point, if we were to have 2 test point, say
$\mathrm{x} = [x_{p}, x_{q}] = [3, 6]$, then if
$h_{\mathcal{D}}(x_{p}) = 4$ and $f(x_{p}) = 6$, then our
$(h_{\mathcal{D}}(\mathrm{x}_{p}) - f(\mathrm{x}_{p}))^2 = 4$.

Then our
$$\mathcal{E}_{\text{out}}(h) =  \mathbb{E}_{\mathrm{x}}\left[(h_{\mathcal{D}}(\mathrm{x}) - f(\mathrm{x}))^2 \right] = \mathbb{E}_{\mathrm{x}_{q}}[[9, 4]] = \frac{1}{2} [9 + 4] = 6.5$$

Note how I secretly removed the subscript in $\mathrm{x}$, and how when
there are two points, we are taking expectation over the 2 points. So if
we have $m$ test points, then the expectation is taken over all the test
points.

Till now, our hypothesis $h$ is fixed over a particular sample set
$\mathcal{D}$. We will now move on to the next concept on **Expected
Generalization Error** (adding a word Expected in front makes a lot of
difference).

### Bias-Variance Decomposition

After decomposing the expected prediction error into reducible and
irreducible error, we can further decompose the reducible error.

Recall the definition of the **bias** of an estimator.

$$
\text{bias}(\hat{\theta}) \triangleq \mathbb{E}\left[\hat{\theta}\right] - \theta
$$

Also recall the definition of the **variance** of an estimator.

$$
\mathbb{V}(\hat{\theta}) = \text{var}(\hat{\theta}) \triangleq \mathbb{E}\left [ ( \hat{\theta} -\mathbb{E}\left[\hat{\theta}\right] )^2 \right]
$$

Using this, we further decompose the reducible error (mean squared
error) into bias squared and variance.

$$
\text{MSE}\left(f(x), \hat{f}(x)\right) = 
\mathbb{E}_{\mathcal{D}} \left[  \left(f(x) - \hat{f}(x) \right)^2 \right] = 
\underbrace{\left(f(x) - \mathbb{E} \left[ \hat{f}(x) \right]  \right)^2}_{\text{bias}^2 \left(\hat{f}(x) \right)} +
\underbrace{\mathbb{E} \left[ \left( \hat{f}(x) - \mathbb{E} \left[ \hat{f}(x) \right] \right)^2 \right]}_{\text{var} \left(\hat{f}(x) \right)}
$$

This is actually a common fact in estimation theory, but we have stated
it here specifically for estimation of some regression function $f$
using $\hat{f}$ at some point $x$.

$$
\text{MSE}\left(f(x), \hat{f}(x)\right) = \text{bias}^2 \left(\hat{f}(x) \right) + \text{var} \left(\hat{f}(x) \right)
$$

In a perfect world, we would be able to find some $\hat{f}$ which is
**unbiased**, that is $\text{bias}\left(\hat{f}(x) \right) = 0$, which
also has low variance. In practice, this isn\'t always possible.

It turns out, there is a **bias-variance tradeoff**. That is, often, the
more bias in our estimation, the lesser the variance. Similarly, less
variance is often accompanied by more bias. Flexible models tend to be
unbiased, but highly variable. Simple models are often extremely biased,
but have low variance.

In the context of regression, models are biased when:

-   Parametric: The form of the model [does not incorporate all the
    necessary
    variables](https://en.wikipedia.org/wiki/Omitted-variable_bias), or
    the form of the relationship is too simple. For example, a
    parametric model assumes a linear relationship, but the true
    relationship is quadratic.
-   Non-parametric: The model provides too much smoothing.

In the context of regression, models are variable when:

-   Parametric: The form of the model incorporates too many variables,
    or the form of the relationship is too flexible. For example, a
    parametric model assumes a cubic relationship, but the true
    relationship is linear.
-   Non-parametric: The model does not provide enough smoothing. It is
    very, \"wiggly.\"

So for us, to select a model that appropriately balances the tradeoff
between bias and variance, and thus minimizes the reducible error, we
need to select a model of the appropriate flexibility for the data.

Recall that when fitting models, we\'ve seen that train RMSE decreases
as model flexibility is increasing. (Technically it is non-increasing.)
For validation RMSE, we expect to see a U-shaped curve. Importantly,
validation RMSE decreases, until a certain flexibility, then begins to
increase.

### Intuitive Explanation of Bias

The error due to bias is taken as the difference between the expected
(or average) prediction of our model and the correct value which we are
trying to predict. Of course you only have one model so talking about
expected or average prediction values might seem a little strange.
However, imagine you could repeat the whole model building process more
than once: each time you gather new data and run a new analysis creating
a new model. Due to randomness in the underlying data sets, the
resulting models will have a range of predictions. Bias measures how far
off in general these models\' predictions are from the correct value.

### Intuitive Explanation of Variance

If you were to be able to rebuild the model process multiples times
producing multiple hypothesis $h$ using multiple datasets $\mathcal{D}$
sampled from the same distribution $\mathcal{P}$, and given a fixed
point $\mathrm{x}_{q}$ which is from the test set (unseen test point),
then can I define (intuitively) the variance of the model over that
fixed point $\mathrm{x}_{q}$ to be:

How much does the prediction of all the $h$ on the test point
$\mathrm{x}_{q}$ , deviate on average, from the mean prediction made by
all the $h$, on that unseen point $\mathrm{x}_{q}$?

### Intuitive Understanding of Bias and Variance

Hi guys, guest here, as usual, please ignore if not important, but I
have been \"stuck\" in the Bias-Variance Decomposition for a week now.
Hope to reaffirm my understanding here.

Consider the general regression setup where we are given a random pair
$(X, Y) \in \mathbb{R}^p \times \mathbb{R}$. We also assume we know the
true/target function $f$ which establishes the true relationship between
$X$ and $Y$. Assume $X$ and the stochastic noise $\epsilon$ is
independent.

With a fixed point $x_{q}$ (the test point is univariate and have one
single point only), can I understand the following:

> **Variance: My understanding**

> > Imagine I built multiple hypothesis $h_{i}$ using multiple datasets
> > $\mathcal{D}_{i}$ sampled from the same distribution $\mathcal{P}$
> > say a uniform distribution $[0, 1]$, and given a fixed point
> > $\mathrm{x}_{q}$ which is from the test set (unseen test point),
> > then can I understand intuitvely that the **variance** of the model
> > over that fixed point $\mathrm{x}_{q}$ to be the **total sum of the
> > average mean squared deviation** of each $h_{i}$ from the average
> > hypothesis $\bar{h}$, where the latter is the mean prediction made
> > by $h_{i}(\mathrm{x}_{q})$, further divided by the number of
> > hypothesis we have (since we are taking two expectations here.

------------------------------------------------------------------------

> **Bias: My understanding**

> > Same setting as above, **bias** of the model over a fixed point
> > $\mathrm{x}_{q}$ to be the **squared error** of $f(\mathrm{x}_{q})$
> > and $\bar{h}(\mathrm{x}_{q})$. In particular, if $\mathrm{x}_{q}$ is
> > has $m$ samples, then we need to sum the squared error of each
> > individual test points and divide by the number of test points.

### HN\'s own notes

For full derivation, you can refer to Learning From Data page 63.

------------------------------------------------------------------------

**There are three sources of error in a model:**

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

$$\begin{align*}
\mathbb{E}_{\mathcal{D}}[\mathcal{E}_{\text{out}}(h)] &= \mathbb{E}_{\mathcal{D}}[\mathbb{E}_{\mathrm{x}}\left[(h_{\mathcal{D}}(\mathrm{x}) - f(\mathrm{x}))^2 \right]] 
\\ &= \big(\;\mathbb{E}_{\mathcal{D}}[\;h_{\mathcal{D}}(x)\;] - f(x)\;\big)^2 + \mathbb{E}_{\mathcal{D}}\big[\;(\;h_{\mathcal{D}}(x) - \mathbb{E}_{\mathcal{D}}[\;h_{\mathcal{D}}(x)\;])^2\;\big] + \mathbb{E}\big[(y-f(x))^2\big]
\\ &= \big(\;\bar{h}(\mathrm{x}) - f(x)\;\big)^2 + \mathbb{E}_\mathcal{D}\big[\;(\;h_{\mathcal{D}}(x) - \bar{h}(\mathrm{x}) \;])^2\;\big]+ \mathbb{E}\big[(y-f(x))^2\big]
\end{align*} 
$$

**Where**:

-   $f(x)$ is the true function of y given the predictors.

-   $h_{\mathcal{D}}(x)$ is the estimate of y with the model fit on a
    random sample $\mathcal{D}_{i}$.

-   $\mathbb{E}_{\mathcal{D}}[\mathbb{E}_{\mathrm{x}}\left[(h_{\mathcal{D}}(\mathrm{x}) - f(\mathrm{x}))^2 \right]]$
    is the average squared error
    across multiple models fit on different random samples between the
    model and the true function. This is also the generalization / test
    error.

-   $\text{E}[\;g(x)\;]$ is the average of estimates for given
    predictors across multiple models fit on different random samples.

-   $E\Big[\big(y - f(x)\big)^2\Big]$ is the average squared error
    between the true values and the predictions from the true function
    of the predictors. This is the **irreducible error**.

-   $\big(\;\mathbb{E}_{\mathcal{D}}[\;h_{\mathcal{D}}(x)\;] - f(x)\;\big)^2$
    is the squared error between the average predictions across multiple
    models fit on different random samples and the prediction of the
    true function. This is the **bias** (squared). **Take a look at my
    notes/simulation to understand it better! Because you should use
    np.sum(np.square(hbar, y_test)) instead of np.square(np.sum(hbar,
    y_test)).**

-   $\mathbb{E}_{\mathcal{D}}\big[\;(\;h_{\mathcal{D}}(x) - \mathbb{E}_{\mathcal{D}}[\;h_{\mathcal{D}}(x)\;])^2\;\big]$
    is the average squared distance between individual model predictions
    and the average prediction of model across multiple random samples.
    This is the **variance**.

-   Average Hypothesis: given size m data points, we draw m number of
    data points $K \to \infty$ of times from our population
    $\mathcal{X} \times \mathcal{Y}$ over a distribution $\mathcal{P}$.
    Each time we draw $m$ number of data points, we will form a sampled
    data $\mathcal{D}_{i}$ where $i = 1,2,3,...,K$, we use our learning
    algorithm $\mathcal{A}$ to learn the parameters using
    $\mathcal{D}_{i}$ and form a hypothesis $h_{i}$ where
    $i = 1,2,3,...,K$. We call the average hypothesis
    $$\bar{h} = \dfrac{1}{K}\sum_{i=1}^{K}h_{i}(x)$$

------------------------------------------------------------------------

-   **The irreducible error is \"noise\" -- error in the measurement of
    our target that cannot be accounted for by our predictors.**
-   The true function represents the most perfect relationship between
    predictors and target, but that does not mean that our variables can
    perfectly predict the target.
-   The irreducible error can be thought of as the measurement error:
    variation in the target that we cannot represent.

------------------------------------------------------------------------

-   **Error due to Bias:** The error due to bias is taken as the
    difference between the expected (or average) prediction of our model
    and the correct value which we are trying to predict. Of course you
    only have one model so talking about expected or average prediction
    values might seem a little strange. However, imagine you could
    repeat the whole model building process more than once: each time
    you gather new data and run a new analysis creating a new model. Due
    to randomness in the underlying data sets, the resulting models will
    have a range of predictions. Bias measures how far off in general
    these models\' predictions are from the correct value.

-   $\big(\;\text{E}[\;g(x)\;] - f(x)\;\big)^2$ is the squared error
    between the average predictions across multiple models fit on
    different random samples and the prediction of the true function.
    This is the **bias** (squared).

------------------------------------------------------------------------

-   **Error due to Variance:** The error due to variance is taken as the
    variability of a model prediction for a given data point. Again,
    imagine you can repeat the entire model building process multiple
    times. The variance is how much the predictions for a given point
    vary between different realizations of the model.

-   Error due to variance is the amount by which the prediction, over
    one training set, differs from the expected value over all the
    training sets. In machine learning, diﬀerent training data sets will
    result in a diﬀerent estimation. But ideally it should not vary too
    much between training sets. However, if a method has high variance
    then small changes in the training data can result in large changes
    in results.

-   $\text{E}\big[\;(\;g(x) - \text{E}[\;g(x)\;])^2\;\big]$ is the
    average squared distance between individual model predictions and
    the average prediction of model across multiple random samples. This
    is the **variance**.

-   Intuitively (but not entirely correct), **Variance** refers to the
    amount by which our $g$ would change if we estimated it using a
    different training dataset. This means, for a ***fixed***
    $g \in \mathcal{H}$, and a training set $\mathcal{D}_{train}$ which
    is also fixed, we resample different
    $\mathcal{D} \in \mathcal{X} \times \mathcal{Y}$, say
    $\mathcal{D}_{i}$ for $i \in [0, \infty]$, and we calculate the
    prediction made for each $g(\mathrm{x}^{(i)})$ and average it to get
    the mean $\text{E}[\;g(\mathrm{x}^{(i)})]$, which is a fixed number.
    Now we take each individual prediction made on the original training
    set $\mathcal{D}_{train}$ and calculate the difference squared from
    the mean. Let us see more details in code.

-   Why did I say not entirely correct, it is because I fixed $g$
    previously, and we will only get $\;(\;g(x) - \text{E}[\;g(x)\;])^2$
    and in order to get the full version, we will need to \"not\" fix
    $g$, and this means, we can have multiple $h_{i}$, and we average
    over all $\;(\;h_{i}(x) - \text{E}[\;h_{i}(x)\;])^2$

-   A more correct way of defining is

> The Bias-Variance Decomposition is done to the prediction error on a
> fixed observation in the test set (only 1 single test/query point).

> We assume we resample our training set again and again and re-train
> the model with each of the resampled train sets.

> For example, the estimation of the error goes in this way: After we
> get $N$ train sets by resampling, we fit $N$ models with each of $N$
> train sets (resampled). With the each of fitted models, we make a
> prediction on the same observation (Out of sample) in the test set.
> With the predictions, we will have $N$ predicted values, and the
> expected value of errors is calculated by taking the average of all
> the prediction errors.

> Now if we are looking at $m$ number of query points, then we have to
> average over $N \times m$ prediction errors!!

> The bias-variance decomposition states that the estimated error
> consists of error from bias, error from variance, and reducible error.

## Bias-Variance Tradeoff

In reality, as the bias goes up, the variance goes down, and vice versa.

------------------------------------------------------------------------

Generally, linear algorithms have a high bias making them fast to learn
and easier to understand but generally less flexible. In turn, they have
lower predictive performance on complex problems that fail to meet the
simplifying assumptions of the algorithms bias.

On the other hands, Variance is the amount that the estimate of the
target function will change if different training data was used.

The target function is estimated from the training data by a machine
learning algorithm, so we should expect the algorithm to have some
variance. Ideally, it should not change too much from one training
dataset to the next, meaning that the algorithm is good at picking out
the hidden underlying mapping between the inputs and the output
variables.

Machine learning algorithms that have a high variance are strongly
influenced by the specifics of the training data. This means that the
specifics of the training have influences the number and types of
parameters used to characterize the mapping function.

### Low and High Bias

-   Low Bias: Suggests less assumptions about the form of the target
    function.

-   High-Bias: Suggests more assumptions about the form of the target
    function.

**Examples of Low and High Bias**

-   Examples of low-bias machine learning algorithms include: Decision
    Trees, k-Nearest Neighbors and Support Vector Machines.

-   Examples of high-bias machine learning algorithms include: Linear
    Regression, Linear Discriminant Analysis and Logistic Regression.

### Low and High Bias {#low-and-high-bias}

-   Low Variance: Suggests small changes to the estimate of the target
    function with changes to the training dataset.

-   High Variance: Suggests large changes to the estimate of the target
    function with changes to the training dataset.

**Examples of Low and High Bias**

Generally, nonlinear machine learning algorithms that have a lot of
flexibility have a high variance. For example, decision trees have a
high variance, that is even higher if the trees are not pruned before
use.

**Examples of Low and High Variance**

-   Examples of low-variance machine learning algorithms include: Linear
    Regression, Linear Discriminant Analysis and Logistic Regression.

-   Examples of high-variance machine learning algorithms include:
    Decision Trees, k-Nearest Neighbors and Support Vector Machines.

### The Tradeoff

The goal of any supervised machine learning algorithm is to achieve low
bias and low variance. In turn the algorithm should achieve good
prediction performance.

You can see a general trend in the examples above:

Linear machine learning algorithms often have a high bias but a low
variance. Nonlinear machine learning algorithms often have a low bias
but a high variance. The parameterization of machine learning algorithms
is often a battle to balance out bias and variance.

Below are two examples of configuring the bias-variance trade-off for
specific algorithms:

-   The k-nearest neighbors algorithm has low bias and high variance,
    but the trade-off can be changed by increasing the value of k which
    increases the number of neighbors that contribute t the prediction
    and in turn increases the bias of the model.
-   The support vector machine algorithm has low bias and high variance,
    but the trade-off can be changed by increasing the C parameter that
    influences the number of violations of the margin allowed in the
    training data which increases the bias but decreases the variance.

------------------------------------------------------------------------

There is no escaping the relationship between bias and variance in
machine learning.

-   Increasing the bias will decrease the variance.

-   Increasing the variance will decrease the bias. There is a trade-off
    at play between these two concerns and the algorithms you choose and
    the way you choose to configure them are finding different balances
    in this trade-off for your problem

In reality, we cannot calculate the real bias and variance error terms
because we do not know the actual underlying target function.
Nevertheless, as a framework, bias and variance provide the tools to
understand the behavior of machine learning algorithms in the pursuit of
predictive performance.

## How to manage Bias and Variance

There are some key things to think about when trying to manage bias and
variance.

### Fight Your Instincts

A gut feeling many people have is that they should minimize bias even at
the expense of variance. Their thinking goes that the presence of bias
indicates something basically wrong with their model and algorithm. Yes,
they acknowledge, variance is also bad but a model with high variance
could at least predict well on average, at least it is not fundamentally
wrong.

This is mistaken logic. It is true that a high variance and low bias
model can preform well in some sort of long-run average sense. However,
in practice modelers are always dealing with a single realization of the
data set. In these cases, long run averages are irrelevant, what is
important is the performance of the model on the data you actually have
and in this case bias and variance are equally important and one should
not be improved at an excessive expense to the other.

### Bagging and Resampling

Bagging and other resampling techniques can be used to reduce the
variance in model predictions. In bagging (Bootstrap Aggregating),
numerous replicates of the original data set are created using random
selection with replacement. Each derivative data set is then used to
construct a new model and the models are gathered together into an
ensemble. To make a prediction, all of the models in the ensemble are
polled and their results are averaged.

One powerful modeling algorithm that makes good use of bagging is Random
Forests. Random Forests works by training numerous decision trees each
based on a different resampling of the original training data. In Random
Forests the bias of the full model is equivalent to the bias of a single
decision tree (which itself has high variance). By creating many of
these trees, in effect a \"forest\", and then averaging them the
variance of the final model can be greatly reduced over that of a single
tree. In practice the only limitation on the size of the forest is
computing time as an infinite number of trees could be trained without
ever increasing bias and with a continual (if asymptotically declining)
decrease in the variance.

### Asymptotic Properties of Algorithms

Academic statistical articles discussing prediction algorithms often
bring up the ideas of asymptotic consistency and asymptotic efficiency.
In practice what these imply is that as your training sample size grows
towards infinity, your model\'s bias will fall to 0 (asymptotic
consistency) and your model will have a variance that is no worse than
any other potential model you could have used (asymptotic efficiency).

Both these are properties that we would like a model algorithm to have.
We, however, do not live in a world of infinite sample sizes so
asymptotic properties generally have very little practical use. An
algorithm that may have close to no bias when you have a million points,
may have very significant bias when you only have a few hundred data
points. More important, an asymptotically consistent and efficient
algorithm may actually perform worse on small sample size data sets than
an algorithm that is neither asymptotically consistent nor efficient.
When working with real data, it is best to leave aside theoretical
properties of algorithms and to instead focus on their actual accuracy
in a given scenario.

### Understanding Over- and Under-Fitting

At its root, dealing with bias and variance is really about dealing with
over- and under-fitting. Bias is reduced and variance is increased in
relation to model complexity. As more and more parameters are added to a
model, the complexity of the model rises and variance becomes our
primary concern while bias steadily falls. For example, as more
polynomial terms are added to a linear regression, the greater the
resulting model\'s complexity will be 3. In other words, bias has a
negative first-order derivative in response to model complexity 4 while
variance has a positive slope.

Bias and variance contributing to total error. ![Bias and variance
contributing to total
error.](https://drive.google.com/uc?id=11ZUNDsLo50flNySlszfNBPx2E1YqNHrr)

Understanding bias and variance is critical for understanding the
behavior of prediction models, but in general what you really care about
is overall error, not the specific decomposition. The sweet spot for any
model is the level of complexity at which the increase in bias is
equivalent to the reduction in variance. Mathematically:

$$\dfrac{dBias}{dComplexity}=−\dfrac{dVariance}{dComplexity}$$

------------------------------------------------------------------------

If our model complexity exceeds this sweet spot, we are in effect
over-fitting our model; while if our complexity falls short of the sweet
spot, we are under-fitting the model. In practice, there is not an
analytical way to find this location. Instead we must use an accurate
measure of prediction error and explore differing levels of model
complexity and then choose the complexity level that minimizes the
overall error. A key to this process is the selection of an accurate
error measure as often grossly inaccurate measures are used which can be
deceptive. The topic of accuracy measures is discussed here but
generally resampling based measures such as cross-validation should be
preferred over theoretical measures such as Aikake\'s Information
Criteria.

## Derivation of Decomposition of Expected Generalized Error

[Link](https://stats.stackexchange.com/questions/164378/bias-variance-decomposition-and-independence-of-x-and-epsilon?rq=1)

Here is a derivation of the bias-variance decomposition, in which I make
use of the independence of $X$ and $\epsilon$.

### True model

Suppose that a target variable $Y$ and a feature variable $X$ are
related via $Y = f(X) + \epsilon$, where $X$ and $\epsilon$ are
independent random variables and the expected value of $\epsilon$ is
zero, $E[\epsilon] = 0$.

We can use this mathematical relationship to generate a data set
$\cal D$. Because data sets are always of finite size, we may think of
$\cal D$ as a random variable, the realizations of which take the form
$d = \{ (x_1,y_1), \ldots , (x_m,y_m) \}$, where $x_i$ and $y_i$ are
realizations of $X$ and $Y$.

### Estimated model

Machine learning uses a particular realization $d$ of $\cal D$ to train
an estimate of the function $f(x)$, called the hypothesis $h_d(x)$. The
subscript $d$ reminds us that the hypothesis is a random function that
varies over training data sets.

### Test error of the estimated model

Having learned an hypothesis for a particular training set $d$, we next
evaluate the error made in predicting the value of $y$ on an unseen test
value $x$. In linear regression, that test error is quantified by taking
a test data set (also drawn from the distribution of $\cal D$) and
computing the average of $(Y - h_d)^2$ over the data set. If the size of
the test data set is large enough, this average is approximated by
$E_{X,\epsilon} [ (Y(X,\epsilon) - h_{d}(X))^2 ]$. As the training data
set $d$ varies, so does the test error; in other words, test error is a
random variable, the average of which over all training sets is given by

\\begin{equation*} \\text{expected test error} = E\_{\\cal D} \\left\[
E\_{X,\\epsilon} \\left\[ (Y(X,\\epsilon) - h\_{\\cal D}(X))\^2
\\right\] \\right\]. \\end{equation*}

In the following sections, I will show how this error arises from three
sources: a *bias* that quantifies how much the average of the hypothesis
deviates from $f$; a *variance* term that quantifies how much the
hypothesis varies among training data sets; and an *irreducible error*
that describes the fact that one\'s ability to predict is always limited
by the noise $\epsilon$.

### Establishing a useful order of integration

To compute the expected test error analytically, we rewrite the
expectation operators in two steps. The first step is to recognize that
\$ E\_{X,\\epsilon} \[\\ldots\] = E_X \\left\[ E\_\\epsilon \[ \\ldots
\] \\right\],\$ **since $X$ and $E$ are independent**. The second step
is to use Fubini\'s theorem to reverse the order in which $X$ and $D$
are integrated out. The final result is that the expected test error is
given by

$$ 
\text{expected test error} = 
E_X \left[ E_{\cal D} \left[ E_\epsilon \left[
(Y - h)^2 
\right] \right] \right], $$

where I have dropped the dependence of $Y$ and $h$ on $X$, $\epsilon$
and $\cal D$ in the interests of clarity.

### Reducible and irreducible error {#reducible-and-irreducible-error}

We fix values of $X$ and $\cal D$ (and therefore $f$ and $h$) and
compute the inner-most integral in the expected test error:

\\begin{align*} E\_\\epsilon \\left\[ (Y - h)\^2 \\right\] & =
E\_\\epsilon \\left\[ (f + \\epsilon - h)\^2 \\right\]\\ & =
E\_\\epsilon \\left\[ (f-h)\^2 + \\epsilon\^2 + 2\\epsilon (f-h)
\\right\]\\ & = (f-h)\^2 + E\_\\epsilon\\left\[ \\epsilon\^2 \\right\] +
0 \\ & = (f-h)\^2 + Var\_\\epsilon \\left\[ \\epsilon \\right\].
\\end{align*}

The last term remains unaltered by subsequent averaging over $X$ and
$D$. It represents the irreducible error contribution to the expected
test error.

The average of the first term,
$E_X \left[ E_{\cal D} \left[ \left( f-h\right)^2 \right] \right]$, is
sometimes called the reducible error.

### Decomposing the reducible error into \'bias\' and \'variance\'

We relax our constraint that $\cal D$ is fixed (but keep the constraint
that $X$ is fixed) and compute the innermost integral in the reducible
error:

$$
\begin{align}
E_{\cal D} \left[ (f-h)^2 \right] 
&= E_{\cal D} \left[ f^2 + h^2 - 2fh \right] \\
&= f^2 + E_{\cal D} \left[ h^2 \right] - 2f E_{\cal D} \left[h\right] \\
\end{align}
$$

Adding and subtracting $E_{\cal D} \left[ h^2 \right]$, and rearranging
terms, we may write the right-hand side above as

$$
\left( f - E_{\cal D} \left[ h \right] \right)^2 + Var_{\cal D} \left[ h \right].
$$

Averaging over $X$, and restoring the irreducible error, yields finally:

$$
\boxed{
\text{expected test error} = 
E_X \left[ \left( f - E_{\cal D} \left[ h \right] \right)^2 \right]
+ E_X \left[ Var_{\cal D} \left[ h \right] \right] 
+ Var_\epsilon \left[ \epsilon \right].
}
$$

The first term is called the bias and the second term is called the
variance.

The variance component of the expected test error is a consequence of
the finite size of the training data sets. In the limit that training
sets contain an infinite number of data points, there are no
fluctuations in $h$ among the training sets and the variance term
vanishes. Put another way, when the size of the training set is large,
the expected test error is expected to be solely due to bias (assuming
the irreducible error is negligible).

### More info

An excellent exposition of these concepts and more can be found
[here](https://www.youtube.com/watch?v=zrEyxfl2-a8).


## References

### Further Readings

- http://scott.fortmann-roe.com/docs/BiasVariance.html
- [STAT 430](https://statisticallearning.org/bias-variance-tradeoff.html) - Very good derivation
- [STAT432 The Bias-Variance Tradeoff](https://statisticallearning.org/bias-variance-tradeoff.html)
- An Introduction to Statistical Learning
    - p34-35
- https://stats.stackexchange.com/questions/164378/bias-variance-decomposition-and-independence-of-x-and-epsilon?rq=1
- https://stats.stackexchange.com/questions/469384/bias-variance-decomposition-expectations-over-what-
- http://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/
