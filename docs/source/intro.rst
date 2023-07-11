.. _introduction:

Introduction
============
The time-dependent adjoint-based gradient is characterised by its backward progression in time and 
its dependence on the solution of a forward system. Consequently, the time-dependent adjoint 
requires either storing the forward data or recomputation of the forward problem to obtain essential 
data for the adjoint computations.

Storing the forward data at every time step is impractical for large systems, as it can result in high
memory usage for extensive time executions. This challenge can be overcome by employing the 
checkpointing method, which is an efficient manages memory usage [3, 5]. In summary, checkpointing strategies 
involve creating a sequence of schedules that coordinate the selective storage of forward data at specific time steps. 
During the adjoint computation, these schedules provide instructions for restarting the forward solver from the nearest 
time step where forward data was stored, in case the forward data is unavailable, until the step where the adjoint 
needs to be computed.


To determine an optimal checkpointing strategy, the revolvers algorithm aims to obtain an optimal 
schedule with a minimal number of forward solver recomputation according the desired checkpoint storage. 
The revolvers algorithm provide an optimal schedule in scenarios where the number of forward steps 
is known prior to the calculation, and where checkpoints store the necessary data for restarting the 
forward computation. Revolvers may also store the forward checkpoint data in different type of storage [1, 2, 3, 4].

The *checkpoint_schedules* is a Python package that offers schedules given by a sequence of actions that provides 
functionalities of forward or adjoint advancement over the entire interval of steps, data storage, and data retrieval.
The *checkpoint_schedules* package explicitly incorporates data buffering in an intermediate storage, ensuring that forward 
variables can be defined and computed before storage in a checkpoint. Furthermore, the schedule distinguishes 
between the storage of forward restart data and the forward data required for the adjoint computation. 
The *checkpoint_schedules* package is flexible to interpret and convert designs from various existing approaches, 
including the revolve algorithm. It has already been successfully with the multistage approach [2], the two-level mixed 
periodic/binomial [6], and H-Revolve schedules [4]. 

To advance in the basics of *checkpoint_schedules* employiment in the adjoint-based gradient computation, 
we recommend to read the following :ref:`section <example_checkpoint_schedules>`.

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

[5] Kukreja, N., Hückelheim, J., Louboutin, M., Washbourne, J., Kelly, P. H., and Gorman, G. J.. 
"Lossy checkpoint compression in full waveform inversion: a case study with ZFPv0. 5.5 and the overthrust model." 
Geoscientific Model Development 15.9 (2022): 3815-3829. DOI: https://doi.org/10.5194/gmd-15-3815-2022.

[6] Aupy, Guillaume, and Julien Herrmann. "Periodicity in optimal hierarchical checkpointing schemes for adjoint computations." 
Optimization Methods and Software 32.3 (2017): 594-624. DOI : https://doi.org/10.1080/10556788.2016.1230612.
