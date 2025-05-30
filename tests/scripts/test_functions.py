# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import typing
from copy import deepcopy
from importlib import import_module

import mithril
from mithril import CBackend, JaxBackend, NumpyBackend, TorchBackend
from mithril.cores.python.numpy.ops_grad import add_grad
from mithril.framework.common import Tensor
from mithril.models import (
    Absolute,
    Add,
    Arange,
    BinaryCrossEntropy,
    Buffer,
    CartesianDifference,
    Cosine,
    CrossEntropy,
    Divide,
    IOKey,
    Layer,
    Linear,
    LinearSVM,
    Mean,
    Model,
    Multiply,
    Power,
    Relu,
    Sigmoid,
    Sine,
    Size,
    Softmax,
    Softplus,
    Subtract,
    TrainModel,
)
from mithril.utils.utils import BiMultiMap
from tests.scripts.test_utils import compare_callables

from ..utils import MyAdder, with_temp_file

# ruff: noqa: F821


def test_bimultimap_1():
    values = [["a", "b", "c"], ["c", "b", "a"], ["a", "a", "a", "a", "a"]]
    keys = ["x", "y", "z"]
    dict_1 = {key: value for key, value in zip(keys, values, strict=False)}
    bi_multi_map_obj: BiMultiMap[str, list[str]] = BiMultiMap(dict_1)
    assert bi_multi_map_obj.inverse == {
        "a": ["x", "y", "z", "z", "z", "z", "z"],
        "b": ["x", "y"],
        "c": ["x", "y"],
    }


def test_bimultimap_2():
    values = [["a", "b", "c"], ["c", "b", "a"], ["a", "a", "a", "a", "a"]]
    keys = ["x", "y", "z"]
    dict_1 = {key: value for key, value in zip(keys, values, strict=False)}
    bi_multi_map_obj: BiMultiMap[str, list[str]] = BiMultiMap(dict_1)
    bi_multi_map_obj_inv: BiMultiMap[str, list[str]] = BiMultiMap(
        bi_multi_map_obj.inverse
    )
    bi_multi_map_obj_inv_inv: BiMultiMap[str, list[str]] = BiMultiMap(
        bi_multi_map_obj_inv.inverse
    )
    table1 = bi_multi_map_obj._table
    table2 = bi_multi_map_obj_inv_inv._table

    table1_inverse = bi_multi_map_obj.inverse
    table2_inverse = bi_multi_map_obj_inv_inv.inverse

    for key, values in table1.items():
        value1 = table2[key]
        value1.sort()
        values.sort()
        assert values == value1

    for key, values in table1_inverse.items():  # type: ignore
        value1 = table2_inverse[key]
        value1.sort()
        values.sort()
        assert values == value1


def test_bimultimap_3():
    values = [["a", "b", "c"], ["c", "b", "a"], ["a", "a", "a", "a", "a"]]
    keys = ["x", "y", "z"]
    remove_item = "x"
    dict_1 = {key: value for key, value in zip(keys, values, strict=False)}
    bi_multi_map_obj: BiMultiMap[str, list[str]] = BiMultiMap(dict_1)
    table1_inv = deepcopy(bi_multi_map_obj.inverse)
    del bi_multi_map_obj[remove_item]
    table2_inv = bi_multi_map_obj.inverse

    for key, values in table1_inv.items():  # type: ignore
        value1 = list(filter(lambda a: a != remove_item, values))
        value2 = table2_inv[key]
        value1.sort()
        value2.sort()
        assert value1 == value2


def test_topological_sort_1():
    linear1 = Linear()
    linear2 = Linear()
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    svm1 = LinearSVM()
    model = Model()

    model |= linear1.connect()
    model |= linear2.connect(input=linear1.output)
    model |= relu1.connect(input=linear2.output)
    model |= relu2.connect(input=relu1.output)
    model |= relu3.connect(input=relu2.output)
    model |= svm1.connect(input=relu3.output, output="output")
    model.expose_keys("output")
    graph = model.get_models_in_topological_order()
    assert graph == [linear1, linear2, relu1, relu2, relu3, svm1]


def test_topological_sort_2():
    relu1 = Relu()
    relu2 = Relu()
    relu3 = Relu()
    relu4 = Relu()
    relu5 = Relu()
    relu6 = Relu()
    model = Model()
    model |= relu1.connect()
    model |= relu2.connect(output=relu1.input)
    model |= relu3.connect(input=relu1.output)
    model |= relu4.connect(input=relu3.output)
    model |= relu5.connect(output=relu2.input)
    model |= relu6.connect(output=relu5.input)
    graph = model.get_models_in_topological_order()
    assert graph == [relu6, relu5, relu2, relu1, relu3, relu4]


def test_topological_sort_3():
    model = Model()
    model1 = Model()
    model2 = Model()
    add1 = Add()
    add2 = Add()
    buff1 = Buffer()
    buff2 = Buffer()
    model1 |= add1.connect(left="input", right="input", output="output")
    model2 |= buff1.connect(input="input", output="output")
    model |= model1.connect(input="input")
    model |= model2.connect(input=model1.output)  # type: ignore
    model |= add2.connect(left=model2.output, right="output")  # type: ignore
    model |= buff2.connect(output=add2.right)
    graph = model.get_models_in_topological_order()
    assert graph == [model1, model2, buff2, add2]


def test_flatten_dag_1():
    model1 = Model()
    model2 = Model()
    model3 = Model()
    model4 = Model()
    add = Add()
    cart = CartesianDifference()
    substract = Subtract()
    mult1 = Multiply()
    power = Power()
    div = Divide()

    ordered_model_list = [add, mult1, cart, substract, power, div]

    model1 |= add.connect(left="in1", right="in2")
    model1 |= mult1.connect(left=add.output, right="in2", output=IOKey(name="output"))

    model2 |= cart.connect(left="in1", right="in2")
    model2 |= substract.connect(left=cart.output, right=cart.output)
    model2 |= power.connect(
        base="in1", exponent=substract.output, output=IOKey(name="output")
    )

    model3 |= div.connect(
        numerator="in1", denominator="in2", output=IOKey(name="output")
    )

    model4 |= model1.connect(in1="input1", in2="input2")
    model4 |= model2.connect(in1=model1.output, in2=model1.output)  # type: ignore
    model4 |= model3.connect(
        in1=model2.output,  # type: ignore
        in2=model2.output,  # type: ignore
        output=IOKey(name="output"),
    )

    comp_model = mithril.compile(
        model=model4, backend=JaxBackend(dtype=mithril.float64), inference=True
    )

    flatted_primitive_model_list = [
        comp_model.flat_graph.connections[key].op.__class__
        for key in comp_model.flat_graph.topological_order
    ]

    assert flatted_primitive_model_list == [
        model.submodel.__class__ for model in ordered_model_list
    ]


def test_flatten_dag_2():
    model1 = Model()
    model2 = Model()
    model3 = Model()
    model4 = Model()

    relu_0 = Relu()
    sigmoid = Sigmoid()
    softmax = Softmax()
    softplus = Softplus()
    relu = Relu()
    leakyrelu = Relu()
    abs = Absolute()
    sine = Sine()
    cosine = Cosine()

    ordered_model_list = [
        relu_0,
        sigmoid,
        sine,
        cosine,
        softmax,
        softplus,
        relu,
        leakyrelu,
        abs,
    ]

    model1 |= relu_0.connect(input="in1")
    model1 |= sigmoid.connect(input="in1", output=IOKey(name="out1"))
    model1 |= softmax.connect(input=relu_0.output, output=IOKey(name="out2"))

    model2 |= softplus.connect(input="in1")
    model2 |= relu.connect(input=softplus.output, output=IOKey(name="out1"))
    model2 |= leakyrelu.connect(input="in2")
    model2 |= abs.connect(input=leakyrelu.output, output=IOKey(name="out2"))

    model3 |= sine.connect(input="in1")
    model3 |= cosine.connect(input=sine.output, output=IOKey(name="out"))

    model4 |= model1.connect(in1="in1")
    model4 |= model3.connect(in1=model1.out1)  # type: ignore
    model4 |= model2.connect(
        in1=model3.out,  # type: ignore
        in2=model1.out2,  # type: ignore
        out1=IOKey(name="out1"),
        out2=IOKey(name="out2"),
    )

    comp_model = mithril.compile(
        model=model4, backend=JaxBackend(dtype=mithril.float64), inference=True
    )

    flatted_primitive_model_list = [
        comp_model.flat_graph.connections[key].op.__class__
        for key in comp_model.flat_graph.topological_order
    ]

    assert flatted_primitive_model_list == [
        model.submodel.__class__ for model in ordered_model_list
    ]


def test_flatten_dag_3():
    model1 = Model()

    relu_0 = Relu()
    sigmoid = Sigmoid()
    softmax = Softmax()
    softplus = Softplus()
    relu = Relu()
    leakyrelu = Relu()
    abs = Absolute()
    sine = Sine()

    model1 |= relu_0.connect(input="in1")
    model1 |= sigmoid.connect(input="in2")
    model1 |= softmax.connect(input="in3")
    model1 |= softplus.connect(input="in4")
    model1 |= relu.connect(input=softplus.output, output=IOKey(name="out4"))
    model1 |= leakyrelu.connect(input=softmax.output, output=IOKey(name="out3"))
    model1 |= abs.connect(input=sigmoid.output, output=IOKey(name="out2"))
    model1 |= sine.connect(input=relu_0.output, output=IOKey(name="out1"))

    ordered_model_list = [
        relu_0,
        sigmoid,
        softmax,
        abs,
        softplus,
        sine,
        relu,
        leakyrelu,
    ]

    comp_model = mithril.compile(
        model=model1, backend=JaxBackend(dtype=mithril.float64), inference=True
    )

    flatted_primitive_model_list = [
        comp_model.flat_graph.connections[key].op.__class__
        for key in comp_model.flat_graph.topological_order
    ]

    assert flatted_primitive_model_list == [
        model.submodel.__class__ for model in ordered_model_list
    ]


@with_temp_file(".py")
def test_code_generator_1(file_path: str):
    model = Model()
    Lin1 = Linear()

    model |= Lin1.connect(input="add1", output=IOKey(name="output"))

    mithril.compile(
        model=model,
        backend=TorchBackend(dtype=mithril.float64),
        jit=False,
        file_path=file_path,
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    @typing.no_type_check
    def evaluate(params, data, cache):
        add1 = data["add1"]
        bias = params["bias"]
        weight = params["weight"]
        output_0 = transpose(weight, None)
        output_1 = matrix_multiplication(add1, output_0)
        del output_0
        output = add(output_1, bias)
        del output_1
        return {"output": output}

    compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_code_generator_2(file_path: str):
    model = Model()
    buff1 = Buffer()
    buff2 = Buffer()
    buff3 = Buffer()
    buff4 = Buffer()

    model |= buff1.connect(input=IOKey("input", type=Tensor), output="output1")
    model |= buff2.connect(input=buff1.output)
    model |= buff3.connect(input=buff1.output)
    model |= buff4.connect(input=buff2.output, output="output2")
    model.expose_keys("output1", "output2")

    mithril.compile(
        model=model,
        backend=TorchBackend(dtype=mithril.float64),
        jit=False,
        file_path=file_path,
        inference=True,
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    def evaluate(params, data, cache):
        input = data["input"]
        return {"output1": input, "output2": input}

    compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_code_generator_3(file_path: str):
    model = Model()
    Linear1 = Linear()
    Linear2 = Linear()

    model |= Linear1.connect(input="input")
    model += Linear2.connect(output=IOKey(name="output"))

    mithril.compile(
        model=model,
        backend=TorchBackend(dtype=mithril.float64),
        jit=False,
        file_path=file_path,
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name).evaluate

    @typing.no_type_check
    def evaluate(params, data, cache):
        bias_0 = params["bias_0"]
        bias_1 = params["bias_1"]
        input = data["input"]
        weight_0 = params["weight_0"]
        weight_1 = params["weight_1"]
        output_3 = transpose(weight_1, None)
        output_0 = transpose(weight_0, None)
        output_1 = matrix_multiplication(input, output_0)
        del output_0
        output_2 = add(output_1, bias_0)
        del output_1
        output_4 = matrix_multiplication(output_2, output_3)
        del output_2
        del output_3
        output = add(output_4, bias_1)
        del output_4
        return {"output": output}

    compare_callables(evaluate, eval_func)


@with_temp_file(".py")
def test_code_generator_4(file_path: str):
    model = Model()

    def my_adder(left, right, cache: None):
        return left + right

    NumpyBackend.register_primitive(my_adder, add_grad)

    model |= MyAdder().connect(left="left", right="right", output="output")
    model.expose_keys("output")
    model.set_differentiability(left=True)
    model.set_differentiability(right=True)

    context = TrainModel(model)
    context.add_loss(
        BinaryCrossEntropy(), reduce_steps=[Mean()], input="output", target="target"
    )
    mithril.compile(
        model=context,
        backend=NumpyBackend(dtype=mithril.float64),
        jit=False,
        file_path=file_path,
        data_keys={"target"},
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name)

    @typing.no_type_check
    def evaluate(params, data, cache):
        left = params["left"]
        output_0_cache = cache["output_0_cache"]
        output_1_cache = cache["output_1_cache"]
        output_cache = cache["output_cache"]
        right = params["right"]
        target = data["target"]
        threshold = cache["threshold"]
        output = output_cache["output"] = make_array(
            my_adder(left, right, output_cache)
        )
        output_0 = output_0_cache["output"] = make_array(
            binary_cross_entropy_with_logits(
                output, target, threshold, cache=output_0_cache
            )
        )
        output_1 = output_1_cache["output"] = make_array(
            reduce_mean(output_0, cache=output_1_cache)
        )
        del output_0
        return {"final_cost": output_1, "output": output}

    @typing.no_type_check
    def evaluate_gradients(params, gradients, data, cache):
        left = params["left"]
        output = cache["output_cache"]["output"]
        output_0 = cache["output_0_cache"]["output"]
        output_0_cache = cache["output_0_cache"]
        output_1_cache = cache["output_1_cache"]
        output_cache = cache["output_cache"]
        right = params["right"]
        target = data["target"]
        threshold = cache["threshold"]
        gradients["output_1"] += gradients["final_cost"]
        gradients["output_0"] += accumulate_grads(
            make_array(
                reduce_mean_grad(gradients["output_1"], output_1_cache, 0, output_0)
            ),
            output_0,
            output_1_cache,
            0,
        )
        gradients["output"] += make_array(
            binary_cross_entropy_with_logits_grad(
                gradients["output_0"], output_0_cache, 0, output, target, threshold
            )
        )
        gradients["left"] += accumulate_grads(
            make_array(add_grad(gradients["output"], output_cache, 0, left, right)),
            left,
            output_cache,
            0,
        )
        gradients["right"] += accumulate_grads(
            make_array(add_grad(gradients["output"], output_cache, 1, left, right)),
            right,
            output_cache,
            1,
        )

    compare_callables(evaluate, eval_func.evaluate)
    compare_callables(evaluate_gradients, eval_func.evaluate_gradients)
    NumpyBackend.registered_primitives = {}


@with_temp_file(".py")
def test_code_generator_5(file_path: str):
    model = Model()

    def my_adder(left, right):
        return left + right

    JaxBackend.register_primitive(my_adder)

    model += MyAdder().connect(left="left", right="right", output="output")
    model.expose_keys("output")
    model.set_differentiability(left=True)
    model.set_differentiability(right=True)

    context = TrainModel(model)
    add = Add()
    add.set_types(right=Tensor)
    add.set_cin("left")
    add.set_differentiability(right=True)

    context.add_loss(
        BinaryCrossEntropy(), reduce_steps=[add], input="output", target="target"
    )
    mithril.compile(
        model=context,
        backend=JaxBackend(dtype=mithril.float64),
        jit=False,
        file_path=file_path,
        data_keys={"target"},
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name)

    @typing.no_type_check
    def evaluate(params, data, cache):
        left = params["left"]
        right = params["right"]
        right_0 = params["right_0"]
        target = data["target"]
        threshold = cache["threshold"]
        output = my_adder(left, right)
        output_0 = binary_cross_entropy_with_logits(output, target, threshold)
        output_1 = add(output_0, right_0)
        del output_0
        return {"final_cost": output_1, "output": output}

    compare_callables(evaluate, eval_func.evaluate)
    JaxBackend.registered_primitives = {}


@with_temp_file(".py")
def test_code_generator_6(file_path: str):
    # Case array creator primitive used in static

    backend = TorchBackend(device="cpu")

    model = Model()
    layer2 = Layer(dimension=2, activation=Softmax())
    model |= layer2.connect(input="input", weight="w1", bias="b1")
    model |= (arange := Arange()).connect(stop=2, output="arange_res")
    model |= Add().connect(left=arange.output, right=layer2.output, output="output")
    model.expose_keys("arange_res", "output")

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )

    static_keys = {"target": backend.array([0])}

    mithril.compile(
        context,
        backend=backend,
        constant_keys=static_keys,  # type: ignore
        jit=False,
        file_path=file_path,
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name)

    @typing.no_type_check
    def evaluate(params, data, cache):
        arange_res = cache["arange_res"]
        b1 = params["b1"]
        input = data["input"]
        target = cache["target"]
        threshold = cache["threshold"]
        w1 = params["w1"]
        output_0 = transpose(w1, None)
        output_1 = matrix_multiplication(input, output_0)
        del output_0
        output_2 = add(output_1, b1)
        del output_1
        output_3 = softmax(output_2)
        del output_2
        output = add(arange_res, output_3)
        del output_3
        output_4 = cross_entropy(output, target, False, threshold)
        output_5 = reduce_mean(output_4)
        del output_4
        return {"arange_res": arange_res, "final_cost": output_5, "output": output}

    compare_callables(evaluate, eval_func.evaluate)
    JaxBackend.registered_primitives = {}


@with_temp_file(".py")
def test_code_generator_7(file_path: str):
    # Case array creator partially initialized

    backend = TorchBackend(device="cpu")

    model = Model()
    layer2 = Layer(dimension=2, activation=Softmax())
    model |= layer2.connect(input="input", weight="w1", bias="b1")
    model += (s := Size(dim=1)).connect()
    model |= (arange := Arange()).connect(stop=s.output, output="arange_res")
    model |= Add().connect(left=arange.output, right=layer2.output, output="output")
    model.expose_keys("arange_res", "output")

    context = TrainModel(model)
    context.add_loss(
        CrossEntropy(input_type="probs"), [Mean()], target="target", input="output"
    )

    static_keys = {"target": backend.array([0])}

    mithril.compile(
        context,
        backend=backend,
        constant_keys=static_keys,  # type: ignore
        jit=False,
        file_path=file_path,
    )

    file_name = os.path.basename(file_path).split(".")[0]
    eval_func = import_module("tmp." + file_name)

    @typing.no_type_check
    def evaluate(params, data, cache):
        arange_res = cache["arange_res"]
        b1 = params["b1"]
        input = data["input"]
        target = cache["target"]
        threshold = cache["threshold"]
        w1 = params["w1"]
        output_0 = transpose(w1, None)
        output_1 = matrix_multiplication(input, output_0)
        del output_0
        output_2 = add(output_1, b1)
        del output_1
        output_3 = softmax(output_2)
        del output_2
        output = add(arange_res, output_3)
        del output_3
        output_5 = cross_entropy(output, target, False, threshold)
        output_6 = reduce_mean(output_5)
        del output_5
        return {"arange_res": arange_res, "final_cost": output_6, "output": output}

    compare_callables(evaluate, eval_func.evaluate)


@with_temp_file(".c")
def test_code_generator_8(file_path: str):
    # Case array creator partially initialized

    backend = CBackend()

    model = Model()
    add = Add()
    add.set_types(left=Tensor, right=Tensor)
    add.set_differentiability(left=True)
    add.set_differentiability(right=True)

    model |= add.connect(left="left", right="right")
    model |= Multiply().connect(left=add.output, right="right2", output="output")
    model.set_types(right2=Tensor)

    mithril.compile(model, backend=backend, jit=False, file_path=file_path)

    code = []
    with open(file_path) as f:
        code = f.readlines()

    eval_code = ""

    start_line = -1
    end_line = -1

    for idx, line in enumerate(code):
        if "evaluate" in line:
            start_line = idx
            break

    for idx, line in enumerate(code[start_line:]):
        if line == "\n":
            end_line = idx
            break

    eval_code = "".join(code[start_line : start_line + end_line])

    evaluate_gradient_code = ""

    start_line = -1
    end_line = len(code)

    for idx, line in enumerate(code):
        if "evaluate_gradients" in line:
            start_line = idx
            break

    evaluate_gradient_code = "".join(code[start_line:end_line])

    reference_eval_code = (
        "struct eval_outputs evaluate(\n\tstruct eval_inputs * inputs\n)\n{\n    "
        "add(inputs->output_0, inputs->left, inputs->right);\n    multiplication(i"
        "nputs->output, inputs->output_0, inputs->right2);\n    struct eval_outputs "
        "output_struct = { .left = inputs->left, .output = inputs->output, .output_0 "
        "= inputs->output_0, .right = inputs->right, .right2 = inputs->right2 };\n    "
        "return output_struct;\n}\n"
    )

    reference_eval_grad_code = (
        "struct eval_grad_outputs evaluate_gradients(\n\tstruct eval_grad_inputs "
        "* inputs\n)\n{\n    multiplication_grad(inputs->output_grad, 0, inputs->"
        "output, inputs->output_0, inputs->right2, inputs->output_0_grad, NULL);\n"
        "    add_grad(inputs->output_0_grad, 0, inputs->output_0, inputs->left, inputs"
        "->right, inputs->left_grad, inputs->right_grad);\n    add_grad(inputs->"
        "output_0_grad, 1, inputs->output_0, inputs->left, inputs->right, inputs->"
        "left_grad, inputs->right_grad);\n    struct eval_grad_outputs output_struct"
        " = { .left_grad = inputs->left_grad, .right_grad = inputs->right_grad };\n    "
        "return output_struct;\n}"
    )

    assert eval_code == reference_eval_code
    assert evaluate_gradient_code == reference_eval_grad_code
    os.remove(file_path.replace(".c", ".so"))
