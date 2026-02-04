from typing import Any, Dict, Optional, Sequence, Tuple, Union

Reward = float
Info = Dict[str, Any]
Action = Union[int, str]
Sample = Tuple[Reward, Info]
InitialEstimates = Optional[Union[float, Sequence[float]]]
