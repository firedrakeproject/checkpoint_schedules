.. _introduction:

Introduction
============

*checkpoint_schedules* is a Python package that provides schedules . The schedules are designed by a sequence 
of *checkpoint_schedules* actions, namely: *Forward*, *Reverse*, *EndForward*, *EndReverse*,
*Move*, and *Copy*. In essence, these actions provide functionalities to time advancing the 
forward and adjoint solvers, checkpoint data storage and retrieval, and indicate the solver 
finalisations. 

This package is designed to be flexible in interpreting different checkpointing strategies. 
It is ready working with the following checkpointing approaches:

* Revolve [1];

* MultiStage checkpointing [2];

* Disk revolve [3];

* Periodic Disk Revolve [4];

* H-Revolve[5];

* Mixed storage of forward restart and adjoint dependency data [6];

* Two-level binomial checkpointing [7] and [8];

*checkpoint_schedules* is able to execute solvers through trivial checkpointing, 
which entails the forward data storage of all time steps without any checkpointing method. 
Furthermore, *checkpoint_schedules* provides an alternative for the cases where no adjoint 
executions are needed, excluding any actions related to forward data storage and retrieval.

*checkpoint_schedules* provides generators enabling easy iteration over the sequence of schedules 
for any above mentioned checkpointing strategies. That allows users access and employ the wished 
checkpointing schedule based on their needs and computational requirements. 

To advance in the basics of *checkpoint_schedules* usage, we recommend to read the following. 
:ref:`section <example_checkpoint_schedules>`.

References
~~~~~~~~~~

### References

[1] Griewank, A., & Walther, A. (2000). Algorithm 799: revolve: an implementation of checkpointing for the reverse or adjoint mode of computational differentiation. ACM Transactions on Mathematical Software (TOMS), 26(1), 19-45., doi: https://doi.org/10.1145/347837.347846

[2] Stumm, P., & Walther, A. (2009). Multistage approaches for optimal offline checkpointing. SIAM Journal on Scientific Computing, 31(3), 1946-1967. https://doi.org/10.1137/080718036

[3] Aupy, G., Herrmann, J., Hovland, P., & Robert, Y. (2016). Optimal multistage algorithm for adjoint computation. SIAM Journal on Scientific Computing, 38(3), C232-C255. DOI: https://doi.org/10.1145/347837.347846.

[4] Aupy, G., & Herrmann, J. (2017). Periodicity in optimal hierarchical checkpointing schemes for adjoint computations. Optimization Methods and Software, 32(3), 594-624. doi: https://doi.org/10.1080/10556788.2016.1230612

[5] Herrmann, J. and Pallez (Aupy), G. (2020). H-Revolve: a framework for adjoint computation on synchronous hierarchical platforms. ACM Transactions on Mathematical Software (TOMS), 46(2), 1-25. DOI: https://doi.org/10.1145/3378672.

[6] Maddison, J. R. (2023). On the implementation of checkpointing with high-level algorithmic differentiation. arXiv preprint arXiv:2305.09568. https://doi.org/10.48550/arXiv.2305.09568.

[7] Pringle, G. C., Jones, D. C., Goswami, S., Narayanan, S. H. K., and  Goldberg, D. (2016). Providing the ARCHER community with adjoint modelling tools for high-performance oceanographic and cryospheric computation. https://nora.nerc.ac.uk/id/eprint/516314.

[8] Goldberg, D. N., Smith, T. A., Narayanan, S. H., Heimbach, P., and Morlighem, M. (2020). Bathymetric Influences on Antarctic Ice‐Shelf Melt Rates. Journal of Geophysical Research: Oceans, 125(11), e2020JC016370. doi: https://doi.org/10.1029/2020JC016370.


