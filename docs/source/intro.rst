.. _introduction:

About checkpoint_schedules
==========================

*checkpoint_schedules* is a Python package has been developed offers a schedule with 
a sequence of actions that provides functionalities of forward or adjoint advancement 
over the entire interval of steps, forward and adjoint data storage and retrieval. 
The schedule explicitly incorporates the buffering of data in an intermediate storage 
of the forward data, ensuring that forward variables can be defined and computed by 
the forward solver before the storage. Furthermore, the schedule can distingue between 
the storage of forward restart data and forward data required in the adjoint computation. 

This package is designed to be flexible in interpreting and converting designs from various existing strategies. 
It is already functioning with the following checkpoint approaches:
* Revolve ; 
* MultiStage checkpointing;
* Disk revolve;
* Periodic Disk Revolve;
* H-Revolve;
* two-level mixed periodic/binomial checkpointing 
* Mixed storage of forward restart and non-linear dependency data;

In addition to the checkpointing schedules mentioned above, checkpoint_schedules also offers two additional 
types of checkpointing schedules for the cases where no checkpointing strategy is used. 
In this case, the forward data is stored every time step, either in RAM or on disk. 
Also, there is a checkpointing schedule available for situations where no adjoint calculation is required.

*checkpoint_schedules* provides generators, which enable easy iteration over the sequence of schedules for any 
above mentioned checkpointing strategies. This allows users to efficiently access and utilize the desired 
checkpointing schedule based on their specific needs and computational requirements. To advance in the basics 
of *checkpoint_schedules* employment and the generator usage, we recommend to read the following 
:ref:`section <example_checkpoint_schedules>`.


References
~~~~~~~~~~

[1] Stumm, Philipp, and Andrea Walther. "Multistage approaches for optimal offline checkpointing." 
SIAM Journal on Scientific Computing 31.3 (2009): 1946--1967. DOI: https://doi.org/10.1137/080718036

[2] Aupy, G., Herrmann, J., Hovland, P., and Robert, Y.. "Optimal multistage algorithm for adjoint computation." 
SIAM Journal on Scientific Computing 38.3 (2016): C232--C255. DOI: https://doi.org/10.1137/15M1019222

[3] Schanen, M., Marin, O., Zhang, H., and Anitescu, M.. "Asynchronous two-level checkpointing scheme for 
large-scale adjoints in the spectral-element solver Nek5000." Procedia Computer Science 80 
(2016): 1147--1158. DOI: https://doi.org/10.1016/j.procs.2016.05.444 .

[4] Herrmann, J. and Pallez (Aupy), G.. "H-Revolve: a framework for adjoint computation on synchronous hierarchical platforms." 
ACM Transactions on Mathematical Software (TOMS) 46.2 (2020): 1-25. DOI: https://doi.org/10.1145/3378672.

[6] Aupy, Guillaume, and Julien Herrmann. "Periodicity in optimal hierarchical checkpointing schemes for adjoint computations." 
Optimization Methods and Software 32.3 (2017): 594-624. DOI : https://doi.org/10.1080/10556788.2016.1230612.
