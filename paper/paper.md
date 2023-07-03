---
title: 'checkpoint_schedules: a checkpointing for adjoint calculations.'
tags:
  - Python
  - Adjoint-based gradients
  - Checkpointing method
# authors:
#   - name: 
#     affiliation: 1
#   - name: 
#     affiliation: 1
#   - name: 
#     affiliation: 2
affiliations:
#  - name: .
#    index: 1
#  - name:  
#    index: 2
date:  July 2023
bibliography: paper.bib

---
# Summary
The *checkpoint_schedules* is a Python package that prescribes the combination of an original forward computation together one adjoint computation. The *checkpoint_schedules* package provides a set of actions that define the forward and adjoint executions. Therefore, it is essential to import the actions from the *checkpoint_schedules* package to ensure proper functionality.

Hence, from the *checkpoint_schedules* library we import the actions used to execute our current adjoint-based gradient with the H-Revolve checkpointing method [2].

# Background

The actions have specific roles in the computation process. Let us describe their functions in more detail:

- The Forward action is responsible for advancing the forward solver by a specified number of steps. It also configures the intermediate checkpoint data storage during the forward computation.

- The Reverse action is used to execute the adjoint solver starting from a initial step (n0) to the step, n1. Additionally,  it clears the forward data that is used in the adjoint solver.

- The Copy action plays an essential role in the checkpointing process. It copies data from a storage level, which can be either RAM or disk, and transfers it to the forward tape. Additionally, it indicates whether this data can be safely deleted from the storage level, freeing up memory resources.

- The actions End_Forward and End_Reverse serve as indicators of the finalization of the forward and reverse solvers, respectively. They signify the completion of the corresponding computation processes.

We also import two additional objects from the *checkpoint_schedules* package: *RevolveCheckpointSchedule* and *StorageLocation*.

The *RevolveCheckpointSchedule* object serves as an iterator, allowing us to iterate over a sequence of the checkpoint schedule actions. It provides a structured and organized way to execute the actions in the desired order.

The *StorageLocation* object is responsible for specifying the locations where the checkpoint data is stored and copied during the computation. The storage locations are: RAM, DISK, TAPE, and NONE.

- RAM is the first level of storage and holds the checkpoint data that is used to restart the forward solver.

- DISK is referred as the second level of storage.

- TAPE refers to the local storage that holds the checkpoint data used as the initial condition for the forward solver. It represents a separate storage location specifically designated for storing this particular type of checkpoint data.

- NONE indicates that there is no specific storage location defined for the checkpoint data. 


* action_forward: This actions is used to advance the forward solver for a step number (s), and to configure the intermediate storage. The forward action given by checkpoint_schedulues package reads Forward(n0, n1, write_ics, write_adj_deps, storage), where n0 and n1 are respectivelly initial and final step, write_ics is used to indicate whether the data used to restart the forward solver, write_adj_deps indicates whether to save the data used in the adjoint computation, and storage indicates the level (either RAM or disk) to store the data. 

* action_reverse: This actions is used to execute the adjoint solver from the start step n0 to the step n1, and to clear the forward data that was used in the adjoint solver. The Reverse action given by checkpoint_schedules package is Reverse(n0, n1, clear_adj_deps), where n0 and n1 are respectivelly initial and final step, clear_adj_deps indicates whether to clear the forward data used in the adjoint computation.

The actions printed above are in general form described as follows:

* *Forward(n0, n1, write_ics, write_adj_deps, storage)*:

    - Executes the forward solver from step *n0* to step *n1*.
    - Write the forward data of step *n0* if *write_ics* is *True*.
    - Indicates whether to store the forward data for the adjoint computation (*write_adj_deps*).
    - Indicate the storage level for the forward data (storage).

* *Reverse(n0, n1, clear_adj_deps)*:

    - Executes the adjoint solver from step *n0* to step *n1*.
    - Clears the adjoint dependencies (*adj_deps*) used in the adjoint computation.

* *Copy(n, from_storage, to_storage, delete)*:

    - Copy the forward data related to step n from one storage location (*from_storage*) to another storage location (*to_storage*).
    - Indicate whether to delete the copied data from the source storage location (delete).

* *EndForward()*:
    - Indicates the finalization of the forward solver.

* *EndReverse()*:

    - Indicate the finalisation of the adjoint solver.

# Statement of Need



# Summary


