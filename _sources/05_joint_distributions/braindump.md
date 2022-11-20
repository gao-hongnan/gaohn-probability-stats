# Brain Dump

- In his page 241-242, he gave intuition, more specifically, my familiar ImageNet example, for why the joint distribution is important.
For example, an image in ImageNet is a drawn from a latent distribution. Each sample
is $x \in \mathbb{R}^{3 \times 224 \times 224}$, where $3$ is the number of channels, $224$ is the height, and $224$ is the width.
So, if we flatten the image, we get a vector of $x \in \mathbb{R}^{150528}$, then the probability of drawing an image is
determined by the joint distribution $\pdfjoint(x_1, x_2, \ldots, x_{150528})$. For example, 
we can imagine that for the car class, the probability of drawing a Ferrari is lower than
the probability of drawing a Toyota, just because a Ferrari is more expensive than a Toyota,
and hence rarer.