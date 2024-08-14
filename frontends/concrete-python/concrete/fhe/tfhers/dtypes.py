"""
Declaration of `TFHERSIntegerType` class.
"""

from functools import partial
from typing import Any, Union

import numpy as np

from ..dtypes import Integer


class TFHERSParams:
    """Crypto parameters used for a tfhers integer."""

    lwe_dimension: int
    glwe_dimension: int
    polynomial_size: int
    pbs_base_log: int
    pbs_level: int

    def __init__(
        self,
        lwe_dimension: int,
        glwe_dimension: int,
        polynomial_size: int,
        pbs_base_log: int,
        pbs_level: int,
    ):
        self.lwe_dimension = lwe_dimension
        self.glwe_dimension = glwe_dimension
        self.polynomial_size = polynomial_size
        self.pbs_base_log = pbs_base_log
        self.pbs_level = pbs_level

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"tfhers_params<lwe_dim={self.lwe_dimension}, glwe_dim={self.glwe_dimension}, "
            f"poly_size={self.polynomial_size}, pbs_base_log={self.pbs_base_log}, "
            f"pbs_level={self.pbs_level}>"
        )

    def __eq__(self, other: Any) -> bool:  # pragma: no cover
        return (
            isinstance(other, self.__class__)
            and self.lwe_dimension == other.lwe_dimension
            and self.glwe_dimension == other.glwe_dimension
            and self.polynomial_size == other.polynomial_size
            and self.pbs_base_log == other.pbs_base_log
            and self.pbs_level == other.pbs_level
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.lwe_dimension,
                self.glwe_dimension,
                self.polynomial_size,
                self.pbs_base_log,
                self.pbs_level,
            )
        )


class TFHERSIntegerType(Integer):
    """
    TFHERSIntegerType (Subclass of Integer) to represent tfhers integer types.
    """

    carry_width: int
    msg_width: int
    params: TFHERSParams

    def __init__(
        self,
        is_signed: bool,
        bit_width: int,
        carry_width: int,
        msg_width: int,
        params: TFHERSParams,
    ):
        super().__init__(is_signed, bit_width)
        self.carry_width = carry_width
        self.msg_width = msg_width
        self.params = params

    def __eq__(self, other: Any) -> bool:  # pragma: no cover
        return (
            isinstance(other, self.__class__)
            and super().__eq__(other)
            and self.carry_width == other.carry_width
            and self.msg_width == other.msg_width
            and self.params == other.params
        )

    def __str__(self) -> str:
        return (
            f"tfhers<{('int' if self.is_signed else 'uint')}"
            f"{self.bit_width}, {self.carry_width}, {self.msg_width}>"
        )

    def encode(self, value: Union[int, np.integer, list, np.ndarray]) -> np.ndarray:
        """Encode a scalar or tensor to tfhers integers.

        Args:
            value (Union[int, np.ndarray]): scalar or tensor of integer to encode

        Raises:
            TypeError: wrong value type

        Returns:
            np.ndarray: encoded scalar or tensor
        """
        bit_width = self.bit_width
        msg_width = self.msg_width
        if isinstance(value, (int, np.integer)):
            value_bin = bin(value)[2:].zfill(bit_width)
            # msb first
            return np.array(
                [int(value_bin[i : i + msg_width], 2) for i in range(0, bit_width, msg_width)]
            )

        if isinstance(value, list):  # pragma: no cover
            try:
                value = np.array(value)
            except Exception:  # pylint: disable=broad-except
                pass  # pragma: no cover

        if isinstance(value, np.ndarray):
            return np.array([self.encode(int(v)) for v in value.flatten()]).reshape(
                value.shape + (bit_width // msg_width,)
            )

        msg = f"can only encode int, np.integer, list or ndarray, but got {type(value)}"
        raise TypeError(msg)

    def decode(self, value: Union[list, np.ndarray]) -> Union[int, np.ndarray]:
        """Decode a tfhers-encoded integer (scalar or tensor).

        Args:
            value (np.ndarray): encoded value

        Raises:
            ValueError: bad encoding

        Returns:
            Union[int, np.ndarray]: decoded value
        """
        bit_width = self.bit_width
        msg_width = self.msg_width
        expected_ct_shape = bit_width // msg_width

        if isinstance(value, list):  # pragma: no cover
            try:
                value = np.array(value)
            except Exception:  # pylint: disable=broad-except
                pass  # pragma: no cover

        if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.integer):
            msg = f"can only decode list of integers or ndarray of integers, but got {type(value)}"
            raise TypeError(msg)

        if value.shape[-1] != expected_ct_shape:
            msg = (
                f"expected the last dimension of encoded value "
                f"to be {expected_ct_shape} but it's {value.shape[-1]}"
            )
            raise ValueError(msg)

        if len(value.shape) == 1:
            # reversed because it's msb first and we are computing powers lsb first
            return sum(v << i * msg_width for i, v in enumerate(reversed(value)))

        cts = value.reshape((-1, expected_ct_shape))
        return np.array([self.decode(ct) for ct in cts]).reshape(value.shape[:-1])


int8 = partial(TFHERSIntegerType, True, 8)
uint8 = partial(TFHERSIntegerType, False, 8)
int16 = partial(TFHERSIntegerType, True, 16)
uint16 = partial(TFHERSIntegerType, False, 16)

int8_2_2 = partial(TFHERSIntegerType, True, 8, 2, 2)
uint8_2_2 = partial(TFHERSIntegerType, False, 8, 2, 2)
int16_2_2 = partial(TFHERSIntegerType, True, 16, 2, 2)
uint16_2_2 = partial(TFHERSIntegerType, False, 16, 2, 2)
