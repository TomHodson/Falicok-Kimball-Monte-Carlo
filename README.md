# Monte Carlo Project ==

## Fermi-Hubbard Model


$$ H = -t \sum_{<ij>,\sigma} c^\dagger_{i,\sigma}c_{j,\sigma} + U \sum_i n_{i \uparrow} n_{i\downarrow}$$

Half filling: one fermion per lattice site

Could consider an extension of this model where the two spin states have different hopping terms:


$$ H = - \sum_{<ij>,\sigma} t_\sigma c_{i,\sigma}c_{j,\sigma} + U \sum_i n_{i \uparrow} n_{i\downarrow}$$

## Falicov-Kimball Model

If we set the hopping term for the down spins to 0:

 $$t_\downarrow = 0$$

we get a model where we have classical down spin particles (f electons) that are immobile and itinerant c electrons that are free to flow around them.

$$ H = - \sum_{<ij>} c^\dagger_ i c_j + U \sum_i c^\dagger_ i c_i n^f_i - \mu \sum_i (c^\dagger_ i c_i  + n^f_i) $$

$$ n^f_i = f^\dagger_i f_i$$

The f electrons are now immobile because the local number operator \(n^f_i\) for the f electrons commutes with the hamiltonian, hence each local occupation of f electrons is conserved over time.

### Half Filling is everyone's favourite

Setting \( \mu = U/2 \) puts us at half filling. Everyone loves half filling because it's where the interesting things happen.

Why is the crossover at \( \mu = U/2 \)? Intuitive argument:  let

$$n = \sum_i (n^c_i + n^f_i) / N$$

with N the total number of lattice sites.

if \(t  = 0\) the eigenbasis is just the product state of the local number states for both species and n is fixed. If  \(\mu \lt U/2\) then the system fills up to n = 1 but no more because it's not favourable to double fill the sites. Once  \(\mu \gt U/2\) it's favourable to double occupy sites and the system fills up the \( n = 2\) state.

### We need some Long range interations

The Pierls argument tells us that in 1D with short range interactions there's never going to be a finite temperature phase transition. So if we want to see one, we had better have some longer range interations, in an ising model you would write:

$$ H_{\text{int}} = V \sum_{ij} \frac{\tau_i \tau_j}{ |i - j|^{\alpha} }$$

You can show (using RG I think) that:

- \( \alpha < d\) where d is the lattice dimension. These are slowly decaying interations where everthing is too tightly coupled and you get non-extensive behaviour which is to say there's no thermodynamic limit for the variables we care about.

- \( \alpha \gt 2 \) For quickly decaying interactions you get back to the short ranged case where the phase transition is at \( T = 0\)

- \( \alpha = 2 \) you get the Kostelitz-Thouless transition.

- \( d \lt \alpha \lt 2 \)  The case we're looking at here will be the intermediate one, which has a phase transition at finite temperature.



This interaction only really makes sense for a ferromagnetic interaction \(V \lt 0\), for an antiferromagnetic one the system is really frustrated. **What actually happens though?**

### Mapping from the FK model to the Ising Model

We're actually only mapping the f electrons to the ising model in order to figure out how to give them an interesting phase transition which might affect the c electrons. Since the local number operators \(n^f_i\) and \(n^f_i\) have eigenvalues 0 or 1, we can do a mapping like:

$$ \tau_i = (2n_i^ f - 1) $$
$$ n_i^f = \frac{1}{2}(\tau_i + 1) $$

which has eigenvalues \(\pm1\) and is can be used as spin 1/2 variables.

The problem is that if this mapping to give a long range ferromagnetic interaction to the FK model, we'll be giving it a preference for either \(n^f_i = 0\) everywhere or  \(n^f_i = 1\) everywhere. What is more interesting is if we give it a charge density wave ground state. The trick is to do a staggered mapping:


$$ \tau_i = (-1)^i (2n_i^ f - 1) $$
$$ n_i^f = \frac{1}{2}((-1)^i \tau_i + 1) $$

Using this mapping, a ferromagnetic long range coupling leads to a FK model whose f electrons undergo a transition from a disordered state to a CDW state at a finite temperature.

$$ H_{\text{int}} = 4V \sum_{ij} \frac{(-1)^{|i-j|}}{ |i - j|^{\alpha} } (n^f_i - 1/2) (n^f_j - 1/2)$$

## Monte Carlo

Usual monte carlo stuff:

$$ Z = Tr e^{-\beta H} $$

Can define the classical part of the hamiltonian as the part which is composed soley of operators that are conserved.


$$ H_{\text{classical}} = -\mu \sum_i n^f_i + H_{\text{int}}$$

$$ H_{\text{non-classical}} = -\mu \sum_i n^c_i  - t \sum_{<ij>} c^\dagger_i c_j + U \sum_i n^c_i n^f_i$$

For a given configuration of the f electrons, call it  \(\{ n^f_i \}\)  the energy from the classical hamiltionian is easy to calculate because all the operators have definite values. For the rest of the hamiltionian we still have to diagonalise and carry out a proper trace.

$$ Z = \sum_{\{ n^f_i \}} e^ {\beta H_{\text{c}}} Tr e^{-\beta H_{\text{nc}}} $$

## Notes
### 29th October
- Could I use perturbation theory to avoid diagonalising the full matrix? By somehow doing an approximate energy and then refining it?
- Don't forget to think about adding termpering
- Add hd5 support

### 30th October
currently working on the setup_mcc, run_mcc, and gather_mcc functions

### 2nd November
- got everything working on my own machine
- had to add a '-ip_no_inlining' compiler flag to get the intel compiler to work with cythonize
- to buld the cython files use:

```
module load anaconda3/personal intel-suite
python setup.py build_ext --inplace
```

- to install the package and enable the commands like run_mcmc use:
```
pip install --editable ./path/to/project
```
