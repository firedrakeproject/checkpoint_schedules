---
title: 'checkpoint_schedules: '
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
date: 20 August 2020
bibliography: paper.bib

---
# Summary
* action_forward: This actions is used to advance the forward solver for a step number (s), and to configure the intermediate storage. The forward action given by checkpoint_schedulues package reads Forward(n0, n1, write_ics, write_adj_deps, storage), where n0 and n1 are respectivelly initial and final step, write_ics is used to indicate whether the data used to restart the forward solver, write_adj_deps indicates whether to save the data used in the adjoint computation, and storage indicates the level (either RAM or disk) to store the data. 

* action_reverse: This actions is used to execute the adjoint solver from the start step n0 to the step n1, and to clear the forward data that was used in the adjoint solver. The Reverse action given by checkpoint_schedules package is Reverse(n0, n1, clear_adj_deps), where n0 and n1 are respectivelly initial and final step, clear_adj_deps indicates whether to clear the forward data used in the adjoint computation.