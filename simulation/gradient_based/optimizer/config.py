from dataclasses import dataclass
import nlopt

@dataclass
class ConfigOptax:
    num_steps = 20
    learning_rate = 1.
@dataclass
class ConfigNlopt:
    MAXEVAL = 50
    FTOL_ABS = 1e-4
    FTOL_REL = 1e-5
    UPPER_BOUNDS = 1
    LOWER_BOUNDS = 0
    OPTIMISER = nlopt.LD_MMA
    i = 0