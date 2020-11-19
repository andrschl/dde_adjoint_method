# dde_adjoint_method
Julia code for interpolating adjoint sensitivity method for DDEs with constant lags [[1]](#1). 

dde_adjoint.jl contains the interpolating adjoint code followed by a short neural ODE example. It uses Differentialequations.jl [[2]](#2) for the DDE solvers. The own implementation is compared to built-in AD sensitivity methods ForwardDiffSensitivity() and ReverseDiffAdjoint(). 

The method assumes autonomous dynamics, constant lags, and currently also a smooth enough transition between initial history and DDE solution. The discontinuities from the adjoint state are passed to the solver by constant_lags.

## References
<a id="1">[1]</a> 
Calver, Jonathan, and Wayne Enright. “Numerical Methods for Computing Sensitivities for ODEs and DDEs.” Numerical Algorithms 74, no. 4 (April 2017): 1101–17. https://doi.org/10.1007/s11075-016-0188-6.

<a id="2">[2]</a> 
Rackauckas, Christopher and Nie, Qing. “Differentialequations.jl--a performant and feature-rich ecosystem for solving differential equations in julia.“ Journal of Open Research Software, 2017
