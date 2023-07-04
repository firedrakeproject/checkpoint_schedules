.. _introduction:

Introduction
============
The *checkpoint_schedules* package provides a set of actions that prescribes the forward and adjoint
executions, e. g., prescribes when to the data related to a forward time-step shall be stored, or when the 
forward/adjoint solver should advance in time. The *checkpoint_schedules* actions are *Forward, Reverse, 
Copy, End_Forward, End_Reverse*. 

The actions have specific roles in the computation process. Let us describe their functions in more detail:

- The *Forward* action is responsible for advancing the forward solver by a specified number of steps. It also configures the intermediate checkpoint data storage during the forward computation.
    - *Forward(n0, n1, write_ics, write_adj_deps, storage)*:
        - Executes the forward solver from step *n0* to step *n1*.
        - Write the forward data of step *n0* if *write_ics* is *True*.
        - Indicates whether to store the forward data for the adjoint computation (*write_adj_deps*).
        - Indicate the storage level for the forward data (storage).

- The *Reverse* action is used to execute the adjoint solver starting from a initial step (n0) to the step, n1. Additionally,  it clears the forward data that is used in the adjoint solver.
    * *Reverse(n0, n1, clear_adj_deps)*:
        - Executes the adjoint solver from step *n0* to step *n1*.
        - Clears the adjoint dependencies (*adj_deps*) used in the adjoint computation.

- The *Copy* action plays an essential role in the checkpointing process. It copies data from a storage level, which can be either RAM or disk, and transfers it to the forward tape. Additionally, it indicates whether this data can be safely deleted from the storage level, freeing up memory resources.
    * *Copy(n, from_storage, to_storage, delete)*:
        - Copy the forward data related to step n from one storage location (*from_storage*) to another storage location (*to_storage*).
        - Indicate whether to delete the copied data from the source storage location (delete).
- The actions *End_Forward* and *End_Reverse* serve as indicators of the finalization of the forward and reverse solvers, respectively. They signify the completion of the corresponding computation processes.
   
Therefore, it is essential to import the actions from the *checkpoint_schedules* package to ensure proper functionality.
Hence, from the *checkpoint_schedules* library we import the actions used to execute our current adjoint-based gradient with the H-Revolve checkpointing method [2].

Additionally, the schedules are designed to manage the number of checkpoints saved in RAM and on disk. This allows for efficient memory usage and storage management. The specific number of checkpoints to be saved in RAM and on disk can be determined based on the requirements and constraints of the computation.

We also import two additional objects from the *checkpoint_schedules* package: *RevolveCheckpointSchedule* and *StorageLocation*.

The *RevolveCheckpointSchedule* object serves as an iterator, allowing us to iterate over a sequence of the checkpoint schedule actions. It provides a structured and organized way to execute the actions in the desired order.

The *StorageLocation* object is responsible for specifying the locations where the checkpoint data is stored and copied during the computation. The storage locations are: RAM, DISK, TAPE, and NONE.

- RAM is the first level of storage and holds the checkpoint data that is used to restart the forward solver.

- DISK is referred as the second level of storage.

- TAPE refers to the local storage that holds the checkpoint data used as the initial condition for the forward solver. It represents a separate storage location specifically designated for storing this particular type of checkpoint data.

- NONE indicates that there is no specific storage location defined for the checkpoint data. 
