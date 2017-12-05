# Python code to fit stochastic models to single cell transcription data

The code implements a version of the Sequential Monte Carlo Approximate Bayesian Computation (SMC ABC) algorithm to fit stochastic markovian models to single cell gene transcription time course data. The algorithm implementation is based on the [Toni et al, 2009](https://www.ncbi.nlm.nih.gov/pubmed/19205079).

Data sets typically consist of multiple measured time courses (60 - 100) of length 250 time points. The original data files will be made available together with a preprint of the corresponding publication.

## Requirements:

Stochastic simulations can be computationally very intensive depending on the parameters. Simulating multiple cells runs in parallel via ipyparallel. In principle it can run on a normal laptop. I used to run it under Linux on a 2011 Dell Latidue laptop with an i7 quad core processor and on a 2012 Macbook Pro. However, the final model fits were run on a workstation computer with a Xeon 10 core processor which sped things up considerably. Fitting a single data set lasted approx 20h on this machine.

- Python 2.7.
- IPython or Jupyter
- ipyparallel
- NumPy
- SciPy
- Matplotlib

Installing and updating the [Anaconda Python distribution](https://www.continuum.io/) should provide all requirements. It might be necessary to manually install ipyparellel via `conda install`.

## Visualisations

Jupyter notebooks include plots and corresponding code to create those plots. Necessary data is included in the respective folders **experimental_data** and **simulations**.
