from typing import Callable, Optional, Type, List
import numpy as np
from numpy.typing import NDArray


ScalarField = Callable[[float, float], float]
TensorField = Callable[[float, float], NDArray[np.float64]]


if __name__ == '__main__':
    class Hihi:
        def __init__(self) -> None:
            pass
    A = Hihi()
    # print(f"{A.__name__ = }")
    # .__class__.__name__
    print(f"{A.__class__.__name__ = }")