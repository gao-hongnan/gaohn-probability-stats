<div align="center">
<h1>Probability & Statistics Documentation</a></h1>
by Hongnan Gao
Sep, 2022
<br>
</div>


<h4 align="center">
  <a href="https://gao-hongnan.github.io/gaohn-probability-stats/intro.html">Documentation</a>
</h4>


## Introduction

This is the documentation for the course [Introduction to Probability for Data Science](https://probability4datascience.com/).

The bottom-up learning takes place in 3 phases:

- **Conceptual Understanding**, where you learn the intuition, concepts and the mathematics behind the algorithms.
More often than not, you might get stuck in this phase. It is important to keep going and not give up. Things
may be clearer later on.
- **Implementation**, where you implement the algorithms from scratch. This is the phase where you get to know
the inner workings of the algorithms. Unfortunately, while knowing how to ***implement*** does not
necessarily mean your ***understanding*** is deep, it is still a good way to learn and build towards
a deeper understanding. It is particularly useful when your models spectacularly fail one day, and 
you know exactly where to look for the bug.
- **Application**, where you apply the algorithms to real-world problems. This is the phase where you
try to connect the dots and see the big picture. 

**Note that the implementation phase may not always be feasible.**

## Workflow

### Installation

```bash
~/gaohn                                $ git clone https://github.com/gao-hongnan/gaohn-probability-stats.git gaohn-probability-stats
~/gaohn                                $ cd gaohn-probability-stats
~/gaohn/gaohn-probability-stats        $ python -m venv <venv_name> && <venv_name>\Scripts\activate 
~/gaohn/gaohn-probability-stats (venv) $ python -m pip install --upgrade pip setuptools wheel
~/gaohn/gaohn-probability-stats (venv) $ pip install -r requirements.txt
~/gaohn/gaohn-probability-stats (venv) $ pip install myst-nb==0.16.0 
```

The reason for manual install of `myst-nb==0.16.0` is because it is not in sync with the current jupyterbook
version, I updated this feature to be able to show line numbers in code cells.

### Building the book

After cloning, you can edit the books source files located in the `content/` directory. 

You run

```bash
~/gaohn (venv) $ jupyter-book build content/
```

to build the book, and

```bash
~/gaohn (venv) $ jupyter-book clean content/
```

to clean the build files.

A fully-rendered HTML version of the book will be built in `content/_build/html/`.