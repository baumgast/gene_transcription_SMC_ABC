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

## Included experimental data

The subfolder `experimental_data` contains time course light intensity measurements of transcription sites in individual cells at different experimental conditions. Conditions are a titration series of eight estrogen concentrations ranging from 0 pM to 1000 pM. Cells were kept at the desired concentration for three days prior to imaging. In addition, one dual allele data obtained at 10 pM estrogen set is included, too as `Data_first.csv` and `Data_second.csv`. The time interval between consecutive measurements is three minutes and each cell was measured at 250 time points.

The data sets `data_ind_10pM.csv` and `data_ind_1000pM.csv` contain measurements of cells that were initially grown without estrogen for two days then were prepared for microscopy. After 50 minutes of imaging the cells received an estrogen signal of 10 or 1000 pM. Those cells were imaged every 1.5 minutes.

Each data set is accompanied by a corresponding mock measurement ot quantify the fluorescence background.

## Visualisations

Jupyter notebooks include plots and corresponding code to create those plots. Necessary data is included in the respective folders **experimental_data** and **simulations**.
