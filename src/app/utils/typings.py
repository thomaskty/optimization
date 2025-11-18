
from app.models.optimizers.base_optimizer import BaseOptimizer
from app.models.optimizers.milp import MILPOptimizer



OPTIMIZER_TYPES = {
    'milp': MILPOptimizer
}