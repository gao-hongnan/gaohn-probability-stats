# Intuition of Continuous Distribution

## Calculus 

The following content is adapted from {cite}`continous_random_variables_mit_1805`.

Conceptually, one should know that the two views of a definite integral:

1. The area under the curve given by $\int_a^b f(x) dx = \text{area under the curve } f(x) \text{ between } a \text{ and } b$;
2. The limit of the sum of the areas of rectangles of width $\Delta x$ and height $f(x)$, where the rectangles are placed next to each other, starting from $x=a$ and ending at $x=b$. Essentially, this means $\int_a^b f(x) dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i) \Delta x$ where $\Delta x = \frac{b-a}{n}$ and $x_i = a + i \Delta x$.

The connection between the two is therefore:

$$
\text{area} \approx \text{sum of rectangle areas} = f(x_1) \Delta x + f(x_2) \Delta x + \cdots + f(x_n) \Delta x = \sum_{i=1}^n f(x_i) \Delta x
$$

And as the width $\Delta x$ of the intervals gets smaller, the approximation becomes better.

```{figure} ../assets/mit1805_integration.png
---
name: mit1805_integration
---
Area is approximately the sum of rectangles. Image credit {cite}`continous_random_variables_mit_1805`.
```

```{admonition} Note
:class: note

As mentioned in {cite}`continous_random_variables_mit_1805`, the interest in integrals
comes primarily from its interpretation as "sum" and to a much lesser extent from
its interpretation as "area".
```



- unit length
- pdf > 1
- P(X=x) = 0
realize 

1. The author emphasized what is the meaning of the PDF of $X$ at a point $x$. 
That can be interpreted as the small rectangle area within a small interval.
He mentioned for each $x$, the value $\pdf(x)$ is the probability per unit length. Which
is easier to interpret, like if the PDF is 2, then it means it is 2 probability per unit length.
This means it is 2 per hour for example, then it must mean the hour is compesentated.
so it means 0.2 per 1/10 hour.

2. So dont confused if u ask why $P(1< X< 2)$ this is diff from the PDF at a point meaning?

3. It means how much probability is concentrated per unit length (d𝒙) near 𝒙, or how dense the probability is near 𝒙.

IT MEANS IT IS THE PROBABILITY PER FOOT AROUND A SUPER SMALL NEIGHBOURHOOD AROUND  x. 
IT DOES NOT MEAN A BIG RANGE. THIS IS JUST ANSWERING THE EXACT QN ON WHAT PDF AT X = x MEANS.
AND THEREFORE WHY THE HIGHER THE f(x) THE MORE PROBABILITY IS CONCENTRATED AROUND X. IT IS BECAUSE
THE PDF IS THE PROBABILITY PER SAY METER AROUND THAT POINT. SO EVEN THO THE POINT IS A DECIMAL SAY
0.1, BUT WHEN YOU CALCULATE f(0.1) = 10, IT MEANS THAT AROUND THAT POINT IT IS 10 PROBABILITY PER METER.


Let's define a thought experiment.

first show why cannot be 0 wrong idea


{numref}`baby_frequency_1`

```{list-table} Baby Frequency Table, binned by 1 kg
:header-rows: 1
:name: baby_frequency_1

* - $x$, Mass (kg)
  - $5 \leq x < 6$
  - $6 \leq x < 7$
  - $7 \leq x < 8$
  - $8 \leq x < 9$
  - $9 \leq x \leq 10$
* - **Frequency**, $f$
  - 20
  - 48
  - 80
  - 36
  - 16
```


## Further Readings

- Chan, Stanley H. "Chapter 4.1. Probability Density Function." In Introduction to Probability for Data Science, 172-180. Ann Arbor, Michigan: Michigan Publishing Services, 2021. 