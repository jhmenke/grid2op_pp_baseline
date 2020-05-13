__all__ = [
    "PandapowerOPFAgent",
    "evaluate",
]

from l2rpn_baselines.PandapowerOPFAgent.PandapowerOPFAgent import PandapowerOPFAgent
from l2rpn_baselines.PandapowerOPFAgent.evaluate import evaluate
"""
In the __init__ file, it is expected to export 3 classes with names that depends on the name you gave to your baseline.
For example, say you chose to write a baseline with the awesome name "XXX" (what an imagination!) you should export
in this __init__.py file:

- `XXX` [**mandatory**] contains the definition of your baseline. It must follow the directives
   given in "Template.py"
- `evaluate` [**mandatory**] contains the script to evaluate the performance of this baseline. It must
  follow the directive in "evaluate.py"
- `train` [**optional**] contains the script to train your baseline. If provided, it must follow
  the directives given in "train.py"
  
See the import above for an example on how to export your scripts properly.
"""
