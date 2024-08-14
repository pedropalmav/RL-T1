# RL-T1
Code for the first assignment of the course IIC3675 - Reinforcement Learning

## Experiment replication
To replicate reported plots, first clone this repo and then run `Main.py`.
Depending on the experiment you're replicating, you need to make some minor changes to `Main.py`, details below.
### General info

We refactor the main loop implementation, wrapping the orgin
```

```
The main loops iterates over `params`, wich is a list of tuples where each tuple contains parameters taken as input by the agent, namely `(epsilon, bias, step_size)`. In other words, it will run an experiment for each parameter tuple in the list.

If `step_size` is set to  0, the algorithm will instantiate a variable step agent. Otherwise, it will use a constant step agent, with $\alpha$ = `step_size`.

### Experiment A  (from book Fig. 2.2)

### Experiment C (from book Fig. 2.3)
