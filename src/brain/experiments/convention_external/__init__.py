"""Does the reported SIGN of consolidation's effect depend on an undeclared
scoring convention, on a public benchmark that is not ours?

F28 refuted both of this project's measurement novelty claims. Four concurrent
papers already measure that consolidation is net-negative, so F23 is replication.
What survived is narrower and methodological: **the crediting convention decides
the sign**. On the frozen synthetic corpus, the same destructive cell reports
+0.061 crediting a merged record generously, -0.146 under the expected
convention 1/(1+a), and -0.303 under the strict one. No source in the prior art
defines this convention; the nearest work flips its own sign through budget
pressure, which is a different lever.

That finding was true only on a corpus we generated to test our own claim. This
experiment re-runs it on LoCoMo, unchanged machinery, adapted data.

Two outcomes, both reportable and both pre-registered:

* **The sign flips on LoCoMo too**, which makes the convention a general lever
  and every concurrent paper's choice of one an undeclared decision.
* **It does not flip**, because LoCoMo's structure makes it immune. Then the
  claim narrows and the structural reason is the finding.

`design.py` holds the pre-registration, frozen before the first number.
"""

from brain.experiments.convention_external.design import (
    PRE_REGISTRATION,
    PRIMARY,
    classify_outcome,
)
from brain.experiments.convention_external.runner import measure

__all__ = ["PRE_REGISTRATION", "PRIMARY", "classify_outcome", "measure"]
