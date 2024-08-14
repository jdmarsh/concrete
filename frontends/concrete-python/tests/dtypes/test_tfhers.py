"""
Tests of `TFHERSIntegerType` data type.
"""

import numpy as np
import pytest

from concrete.fhe import tfhers

DEFAULT_TFHERS_PARAM = tfhers.TFHERSParams(
    909,
    1,
    4096,
    15,
    2,
)


def parameterize_partial_dtype(partial_dtype) -> tfhers.TFHERSIntegerType:
    """Create a tfhers type from a partial func missing tfhers params.

    Args:
        partial_dtype (callable): partial function to create dtype (missing params)

    Returns:
        tfhers.TFHERSIntegerType: tfhers type
    """

    return partial_dtype(DEFAULT_TFHERS_PARAM)


def test_tfhers_encode_bad_type():
    """Test encoding of unsupported type"""
    dtype = parameterize_partial_dtype(tfhers.uint16_2_2)
    with pytest.raises(TypeError, match=r"can only encode int or ndarray, but got <class 'str'>"):
        dtype.encode("bad type")


def test_tfhers_encode_ndarray():
    """Test ndarray encoding"""
    dtype = parameterize_partial_dtype(tfhers.uint16_2_2)
    shape = (4, 5)
    value = np.random.randint(0, 2**10, size=shape)
    encoded = dtype.encode(value)
    decoded = dtype.decode(encoded)
    assert (decoded == value).all()
    assert encoded.shape == shape + (8,)


def test_tfhers_bad_decode():
    """Test decoding of bad values"""
    dtype = parameterize_partial_dtype(tfhers.uint8_2_2)
    shape = (2, 10)
    bad_value = np.random.randint(0, 2**10, size=shape)
    with pytest.raises(
        ValueError,
        match=r"bad encoding",
    ):
        dtype.decode(bad_value)


def test_tfhers_integer_bad_values():
    """Test new integer with bad values"""
    dtype = parameterize_partial_dtype(tfhers.uint8_2_2)
    with pytest.raises(
        ValueError,
    ):
        tfhers.TFHERSInteger(
            dtype,
            [
                [1, 2],
                [
                    2,
                ],
            ],
        )

    with pytest.raises(
        ValueError,
        match=r"ndarray value has bigger elements than what the dtype can support",
    ):
        tfhers.TFHERSInteger(
            dtype,
            [
                [1, 2],
                [2, 2**10],
            ],
        )

    with pytest.raises(
        ValueError,
        match=r"ndarray value has smaller elements than what the dtype can support",
    ):
        tfhers.TFHERSInteger(
            dtype,
            [
                [1, -2],
                [2, 2],
            ],
        )

    with pytest.raises(
        TypeError,
        match=r"value can either be an int or ndarray, not a <class 'str'>",
    ):
        tfhers.TFHERSInteger(dtype, "bad value")
