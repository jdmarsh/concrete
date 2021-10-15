"""Test file for intermediate representation"""

from copy import deepcopy

import numpy
import pytest

from concrete.common.data_types.floats import Float
from concrete.common.data_types.integers import Integer
from concrete.common.representation import intermediate as ir
from concrete.common.values import ClearScalar, ClearTensor, EncryptedScalar, EncryptedTensor


@pytest.mark.parametrize(
    "node,input_data,expected_result",
    [
        pytest.param(
            ir.Add([EncryptedScalar(Integer(64, False)), EncryptedScalar(Integer(64, False))]),
            [10, 4589],
            4599,
            id="Add",
        ),
        pytest.param(
            ir.Sub([EncryptedScalar(Integer(64, False)), EncryptedScalar(Integer(64, False))]),
            [10, 4589],
            -4579,
            id="Sub",
        ),
        pytest.param(
            ir.Mul([EncryptedScalar(Integer(64, False)), EncryptedScalar(Integer(64, False))]),
            [10, 4589],
            45890,
            id="Mul",
        ),
        pytest.param(ir.Input(ClearScalar(Integer(32, True)), "in", 0), [42], 42, id="Input"),
        pytest.param(ir.Constant(42), None, 42, id="Constant"),
        pytest.param(ir.Constant(-42), None, -42, id="Constant"),
        pytest.param(
            ir.UnivariateFunction(
                EncryptedScalar(Integer(7, False)), lambda x: x + 3, Integer(7, False)
            ),
            [10],
            13,
            id="UnivariateFunction, x + 3",
        ),
        pytest.param(
            ir.UnivariateFunction(
                EncryptedScalar(Integer(7, False)),
                lambda x, y: x + y,
                Integer(7, False),
                op_kwargs={"y": 3},
            ),
            [10],
            13,
            id="UnivariateFunction, (x, y) -> x + y, where y is constant == 3",
        ),
        pytest.param(
            ir.UnivariateFunction(
                EncryptedScalar(Integer(7, False)),
                lambda x, y: y[x],
                Integer(7, False),
                op_kwargs={"y": (1, 2, 3, 4)},
            ),
            [2],
            3,
            id="UnivariateFunction, (x, y) -> y[x], where y is constant == (1, 2, 3, 4)",
        ),
        pytest.param(
            ir.UnivariateFunction(
                EncryptedScalar(Integer(7, False)),
                lambda x, y: y[3],
                Integer(7, False),
                op_kwargs={"y": (1, 2, 3, 4)},
            ),
            [2],
            4,
            id="UnivariateFunction, x, y -> y[3], where y is constant == (1, 2, 3, 4)",
        ),
        pytest.param(
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
            ),
            [[1, 2, 3, 4], [4, 3, 2, 1]],
            20,
            id="Dot, [1, 2, 3, 4], [4, 3, 2, 1]",
        ),
        pytest.param(
            ir.Dot(
                [
                    EncryptedTensor(Float(32), shape=(4,)),
                    ClearTensor(Float(32), shape=(4,)),
                ],
                Float(32),
            ),
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            20,
            id="Dot, [1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]",
        ),
        pytest.param(
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
                delegate_evaluation_function=numpy.dot,
            ),
            [
                numpy.array([1, 2, 3, 4], dtype=numpy.int32),
                numpy.array([4, 3, 2, 1], dtype=numpy.int32),
            ],
            20,
            id="Dot, np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])",
        ),
    ],
)
def test_evaluate(
    node: ir.IntermediateNode,
    input_data,
    expected_result: int,
):
    """Test evaluate methods on IntermediateNodes"""
    assert node.evaluate(input_data) == expected_result


@pytest.mark.parametrize(
    "node1,node2,expected_result",
    [
        (
            ir.Add([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            ir.Add([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            True,
        ),
        (
            ir.Add([EncryptedScalar(Integer(16, False)), EncryptedScalar(Integer(32, False))]),
            ir.Add([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(16, False))]),
            True,
        ),
        (
            ir.Add([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            False,
        ),
        (
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            True,
        ),
        (
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(16, False))]),
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(16, False))]),
            True,
        ),
        (
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(16, False))]),
            ir.Sub([EncryptedScalar(Integer(16, False)), EncryptedScalar(Integer(32, False))]),
            False,
        ),
        (
            ir.Mul([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            ir.Mul([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            True,
        ),
        (
            ir.Mul([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            False,
        ),
        (
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            ir.Sub([EncryptedScalar(Integer(32, False)), EncryptedScalar(Integer(32, False))]),
            False,
        ),
        (
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            True,
        ),
        (
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            ir.Input(EncryptedScalar(Integer(32, False)), "y", 0),
            False,
        ),
        (
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 1),
            False,
        ),
        (
            ir.Input(EncryptedScalar(Integer(32, False)), "x", 0),
            ir.Input(EncryptedScalar(Integer(8, False)), "x", 0),
            False,
        ),
        (
            ir.Constant(10),
            ir.Constant(10),
            True,
        ),
        (
            ir.Constant(10),
            ir.Input(EncryptedScalar(Integer(8, False)), "x", 0),
            False,
        ),
        (
            ir.Constant(10),
            ir.Constant(10.0),
            False,
        ),
        (
            ir.UnivariateFunction(
                EncryptedScalar(Integer(8, False)), lambda x: x, Integer(8, False)
            ),
            ir.UnivariateFunction(
                EncryptedScalar(Integer(8, False)), lambda x: x, Integer(8, False)
            ),
            True,
        ),
        (
            ir.UnivariateFunction(
                EncryptedScalar(Integer(8, False)),
                lambda x: x,
                Integer(8, False),
                op_args=(1, 2, 3),
            ),
            ir.UnivariateFunction(
                EncryptedScalar(Integer(8, False)), lambda x: x, Integer(8, False)
            ),
            False,
        ),
        (
            ir.UnivariateFunction(
                EncryptedScalar(Integer(8, False)),
                lambda x: x,
                Integer(8, False),
                op_kwargs={"tuple": (1, 2, 3)},
            ),
            ir.UnivariateFunction(
                EncryptedScalar(Integer(8, False)), lambda x: x, Integer(8, False)
            ),
            False,
        ),
        (
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
                delegate_evaluation_function=numpy.dot,
            ),
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
                delegate_evaluation_function=numpy.dot,
            ),
            True,
        ),
        (
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
                delegate_evaluation_function=numpy.dot,
            ),
            ir.Dot(
                [
                    EncryptedTensor(Integer(32, True), shape=(4,)),
                    ClearTensor(Integer(32, True), shape=(4,)),
                ],
                Integer(32, True),
            ),
            False,
        ),
    ],
)
def test_is_equivalent_to(
    node1: ir.IntermediateNode,
    node2: ir.IntermediateNode,
    expected_result: bool,
    test_helpers,
):
    """Test is_equivalent_to methods on IntermediateNodes"""
    assert (
        test_helpers.nodes_are_equivalent(node1, node2)
        == test_helpers.nodes_are_equivalent(node2, node1)
        == expected_result
    )


@pytest.mark.parametrize(
    "list_to_fill,expected_list",
    [
        pytest.param([None, 1, 2, 3, None, None], [1, 1, 2, 3, 3, 3]),
        pytest.param([None], None, marks=pytest.mark.xfail(strict=True)),
        pytest.param([None, None, None, None, 7, None, None, None], [7, 7, 7, 7, 7, 7, 7, 7]),
        pytest.param([None, None, 3, None, None, None, 2, None], [3, 3, 3, 3, 3, 2, 2, 2]),
    ],
)
def test_flood_replace_none_values(list_to_fill: list, expected_list: list):
    """Unit test for flood_replace_none_values"""

    # avoid modifying the test input
    list_to_fill_copy = deepcopy(list_to_fill)
    ir.flood_replace_none_values(list_to_fill_copy)

    assert all(value is not None for value in list_to_fill_copy)
    assert list_to_fill_copy == expected_list
