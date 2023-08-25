__version__ = "2.0.3"

"""
By importing transformers before other libs, the segment fault that happens in KGRL docker will
be repaired. In fact, switching the order of modules and optimizer also helps.
"""

import transformers  # noqa
import antmmf.modules
import antmmf.optimizer
import antmmf.utils
import antmmf.models
import antmmf.predictors
import antmmf.tasks
import antmmf.trainers
