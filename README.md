# Physics codes
### Codes writted during my PhD in the [group of Andreas Läuchli](https://www.psi.ch/en/ltc/computergestutzte-physik) at the University of Innsbruck and the Paul Scherrer Institute.

## About
The codes are mainly related to my research in the are of quantum dynamics and condensed matter physics. Most codes are associated with research articles and the corresponding data is available in Zenodo archives. Apart from the code for computations, the code for creating the figures in the articles is also provided The repo consists of

- ***codebase***: folder containing basic codes used across projects or regularly within a project; in particular it includes a Krylov subspace algorithm to compute the matrix exponential (and possible other matrix functions) based on the Lanczos method with [partial reorthogonalization](https://www.jstor.org/stable/2007563).
- ***floquet***: folder containing code for various computations on periodically driven (Floquet) quantum systems. The focus of the codes lies in computing the heating rate of a Floquet system; for details see [Phys. Rev. Research 4, 043174](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.043174) and the corresponding [archive](https://zenodo.org/records/4058928).
- ***annealing***: folder containing code for performing quantum annealing, simulated annealing and simulated quantum annealing on the Sherrington-Kirkpatrick model (can be easily adapted for other models). The goal is to compare the methods for different annealing times (diabatic annealing); for details see [Phys. Rev. A 107, 062602](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.062602) and the corresponding [archive](https://zenodo.org/records/7998615).
- ***nonreciprocal_dipoles***: folder containing code for computing the (nonreciprocal) interactions and the dynamics of a pair of dipoles close to a moving conductor. For details see [Phys. Rev. B 106, 174435](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.174435) and the corresponding [archive](https://zenodo.org/records/7389523).
- ***counterdiabatic_driving***: folder containing code for various computations for counterdiabatic driving using the variational approach. This is based on work done in collaboration with [Anatoli Polkovnikov](http://physics.bu.edu/~asp28/) during a three-month research stay at Boston University; the results are unpublished, but for general details on the variational approach see this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0370157317301989?via%3Dihub).


## Software
- Python
- Scientific Python (NumPy, SciPy, Matplotlib, Numba)
- Mathematica

## Numerical methods
- Krylov subspace methods
- Magnus expansion (also commutator-free version)
- Adaptive integration for highly oscillatory integrals based on Romberg integration
- Various data analysis techniques for large numerical datasets, often tailored to the problem and based on physical understanding.
- Visualization

## Acknowledgements
The code presented has predominantly been written by myself, however, I would like to acknowledge helpful discussions about the physics and/or the numerical methods (and their implementations) with Andreas Läuchli, Anatoli Polkovnikov, Jonathan Wurtz, Oriol Romero-Isart, and Patrick Maurer.
