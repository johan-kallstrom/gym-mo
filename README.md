Status: Under development (code is provided as-is, interfaces may break)

# Gym-MO

Gym-MO provides a number of simulation environments for Multi-Objective Reinforcement Learning (MORL). The environments implement the OpenAI Gym interface, with modification to allow for preferences among objectives and vector valued rewards, according to the Multi-Objective Markov Decision Process (MOMDP).

# Environments

## Gridworlds

### Gathering Environment
In this environment the agent should collect (green, red and yellow) items on the grid and either cooperate or compete with another hard-coded agent collecting red items.

Can use pixel or vector obsevations (by setting from_pixels argument). Set the agent's preferences among objectives in constructor and reset function so that episodes end when no more reward can be gathered.

See: http://www.diva-portal.org/smash/get/diva2:1362933/FULLTEXT01.pdf

### Traffic Environment
In this environment the agent should collect two items in the grid, and must balance the time spent against the risk of colliding with vehicles or braking traffic rules.

Can use pixel or vector obsevations (by setting from_pixels argument). Set the agent's preferences among objectives in constructor and reset function so that episodes end when no more reward can be gathered.

See: http://www.diva-portal.org/smash/get/diva2:1362933/FULLTEXT01.pdf

### Deep Sea Treasure Environment
In this environment the agent operates a submarine in search of treasure on the seabed, and must balance the value of collected treasures against the time spent.

See: https://link.springer.com/content/pdf/10.1007/s10994-010-5232-5.pdf

## Classic Control

### Multi-Objective Mountain Car Environment
In this environment the agent must get an under powered car to the top of a hill, while balancing time spent against the number of braking and acceleration commands.

See: https://link.springer.com/content/pdf/10.1007/s10994-010-5232-5.pdf

# Installation

You can perform an install of ``gym-mo`` with:

```
git clone https://github.com/johan-kallstrom/gym-mo.git
cd gym-mo
pip install -e .
```

Test the installation by running a random agent on the basic gridworld:

```
python gym_mo/envs/gridworlds/gridworld_base.py
```

# Test cases
Test cases can be run by:

```
python gym_mo/test/gridworld_tests.py
```

# Commit conventions
The following prefixes shall be used for commits:

Feature: Used when adding new functionality to the code.
Fix: Used when fixing a bug or other issue in the existing code.
Maintenance: Used for misc modifications of the repo.
Documentation: Used for documentation, e.g. comments in the code or updates of this README.
