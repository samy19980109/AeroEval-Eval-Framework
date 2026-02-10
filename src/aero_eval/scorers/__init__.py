"""Scorer package. Importing submodules triggers auto-registration."""

from aero_eval.scorers import l1_rules  # noqa: F401
from aero_eval.scorers import l2_statistical  # noqa: F401
from aero_eval.scorers import l3_llm_judge  # noqa: F401
from aero_eval.scorers import l4_system  # noqa: F401
from aero_eval.scorers import numerical  # noqa: F401
from aero_eval.scorers import rag  # noqa: F401
