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

import sys
from copy import deepcopy
from typing import Literal

import numpy as np
import pytest

import mithril as ml
from mithril import IOKey
from mithril.framework.physical.model import FlatModel
from mithril.models import (
    Add,
    Arange,
    AtLeast1D,
    Buffer,
    Concat,
    Convolution2D,
    Divide,
    GroupNorm,
    Linear,
    Model,
    Multiply,
    Power,
    Reshape,
    ScaledDotProduct,
    Square,
    Tensor,
    ToList,
    Transpose,
    Where,
    functional,
)

from .helper import assert_models_equal
from .test_utils import normalize_flat_graph


class ComparisonTestBase:
    backend = ml.JaxBackend()

    skip_flat_graph_assertion: Literal[False] | tuple[Literal[True], str] = False
    skip_logical_assertion: Literal[False] | tuple[Literal[True], str] = False

    def model1(self) -> Model:
        # generally for models created by Model.create API
        raise NotImplementedError("This method should be overridden in subclasses")

    def model2(self) -> Model:
        # Generally for models created by model.connect API
        raise NotImplementedError("This method should be overridden in subclasses")

    def test_assert_flat_graphs(self):
        if self.skip_flat_graph_assertion:
            pytest.skip(self.skip_flat_graph_assertion[1])

        model1 = self.model1()
        model2 = self.model2()

        pm1 = ml.compile(
            backend=self.backend, model=model1, inference=True, safe_names=False
        )

        pm2 = ml.compile(
            backend=self.backend, model=model2, inference=True, safe_names=False
        )

        pm1_graph = pm1.flat_graph
        pm2_graph = pm2.flat_graph

        normalized_graph1 = normalize_flat_graph(pm1_graph)
        normalized_graph2 = normalize_flat_graph(pm2_graph)

        assert normalized_graph1 == normalized_graph2

    def test_assert_logical_models(self):
        if self.skip_logical_assertion:
            pytest.skip(self.skip_logical_assertion[1])

        model1 = self.model1()
        model2 = self.model2()

        assert_models_equal(model1, model2)


class CompareEvaluateTestBase(ComparisonTestBase):
    def test_assert_evaluations(self):
        model1 = self.model1()
        model2 = self.model2()

        pm1 = ml.compile(
            backend=self.backend,
            model=model1,
            trainable_keys=[key for key in model1.input_keys if key[0] != "$"],
            inference=True,
            safe_names=True,
        )

        pm2 = ml.compile(
            backend=self.backend,
            model=model2,
            trainable_keys=[key for key in model2.input_keys if key[0] != "$"],
            inference=True,
            safe_names=True,
        )

        params = pm1.randomize_params()

        output1 = pm1.evaluate(params, {})
        output2 = pm2.evaluate(params, {})

        for value1, value2 in zip(output1.values(), output2.values(), strict=False):
            _value1 = np.array(value1)
            _value2 = np.array(value2)
            np.testing.assert_allclose(_value1, _value2)


class TestSimpleModelWithValues(CompareEvaluateTestBase):
    def model1(self):
        key1 = IOKey("key1", shape=[3, 4, 5])
        key2 = IOKey("key2", shape=[3, 4, 5])

        output = (key1 + key2 * 3) / 2
        return Model.create(output=output)

    def model2(self):
        key1 = IOKey("key1", shape=[3, 4, 5])
        key2 = IOKey("key2", shape=[3, 4, 5])

        model = Model()
        model |= Multiply(right=3).connect(key2)
        model |= Add().connect(key1, model.cout)
        model |= Divide(denominator=2).connect(model.cout, output=IOKey("output"))
        model._freeze()

        return model


class TestTwoLinears(CompareEvaluateTestBase):
    def model1(self):
        linear1 = Linear(dimension=4)
        linear2 = Linear(dimension=7)

        input = IOKey("input", shape=[3, 4, 5])

        output = linear1(input=input)
        output = linear2(input=output)

        return Model.create(output=output)

    def model2(self):
        model = Model()

        linear1 = Linear(dimension=4)
        linear2 = Linear(dimension=7)

        input = IOKey("input", shape=[3, 4, 4])

        model |= linear1.connect(input=input)
        model |= linear2.connect(input=model.cout, output=IOKey("output"))
        model._freeze()

        return model


class TestExtendAndExtraction(ComparisonTestBase):
    def model1(self):
        input1 = IOKey()
        input2 = IOKey()
        input3 = IOKey()
        mult_output = input1 * input2
        output = mult_output + input3
        assert output.model is not None
        return output.model

    def model2(self):
        model = Model()
        model |= (mult := Multiply())
        model |= Add().connect(left=mult.output)
        return model


class TestExtendTwoConnections(ComparisonTestBase):
    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        output = input1 * input2
        assert output.model is not None
        return output.model

    def model2(self):
        model = Model()
        model += Multiply().connect(left="input1", right="input2")
        return model


class TestExtendAndExtractionNamed(ComparisonTestBase):
    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        input3 = IOKey("input3")
        mult_output = input1 * input2
        output = mult_output + input3
        assert output.model is not None
        return output.model

    def model2(self):
        model = Model()
        model |= (mult := Multiply()).connect(left="input1", right="input2")
        model |= Add().connect(left=mult.output, right="input3")
        return model


class TestExtendAndExtractionViaExtendApi(ComparisonTestBase):
    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        input3 = IOKey("input3")
        mult_output = input1 * input2
        model = Model()
        model |= Add().connect(left=mult_output, right=input3)
        return model

    def model2(self):
        model = Model()
        model |= (mult := Multiply()).connect(left="input1", right="input2")
        model |= Add().connect(left=mult.output, right="input3")
        return model


class TestExtendConnectionWithModel(ComparisonTestBase):
    def model1(self):
        add = Add()
        input1 = IOKey()
        output = add.output * input1
        assert output.model is not None
        return output.model

    def model2(self):
        model = Model()
        model |= (add2 := Add())
        model |= Multiply().connect(add2.output)
        return model


class TestExtendMultipleModels(ComparisonTestBase):
    def model1(self):
        add = Add()
        input1 = IOKey()
        add.output * input1

        input2 = IOKey()
        output2 = add.output * input2
        assert output2.model is not None
        return output2.model

    def model2(self):
        model = Model()
        model |= (add2 := Add())
        model |= Multiply().connect(add2.output)
        model |= Multiply().connect(add2.output)
        return model


class TestExtendToModelConnectionNested(ComparisonTestBase):
    def model1(self):
        add = Add()
        m1 = Model()
        m1 |= add
        m2 = Model()
        m2 |= m1
        m3 = Model()
        m3 |= m2

        input1 = IOKey()
        output = add.output * input1
        model = Model()
        model |= m3
        model |= Buffer().connect(output)
        return model

    def model2(self):
        add = Add()
        m1 = Model()
        m1 |= add
        m2 = Model()
        m2 |= m1
        m3 = Model()
        m3 |= m2

        model = Model()
        model |= m3
        model |= (mult := Multiply()).connect(add.output)
        model |= Buffer().connect(mult.output)
        return model


class TestExtendAndExtractionSameInputs(ComparisonTestBase):
    def model1(self):
        input1 = IOKey()
        input2 = IOKey()
        add_output = input1 + input2
        mult_output = input1 * input2
        assert add_output.model == mult_output.model == input1.model == input2.model
        return mult_output.model

    def model2(self):
        _input1 = IOKey()
        _input2 = IOKey()

        model = Model()
        model |= Add().connect(left=_input1, right=_input2)
        model |= Multiply().connect(left=_input1, right=_input2)
        return model


class TestExtendExtractionFrozenModels(ComparisonTestBase):
    def model1(self):
        add_output = Add().output * Add().output
        mult_output = Add().output * Add().output
        output = add_output + mult_output
        assert output.model is not None
        return output.model

    def model2(self):
        model = Model()
        model |= (add1 := Add())
        model |= (add2 := Add())
        model |= (mult1 := Multiply()).connect(left=add2.output, right=add1.output)
        model |= (add3 := Add())
        model |= (add4 := Add())
        model |= (mult2 := Multiply()).connect(left=add4.output, right=add3.output)
        model |= Add().connect(left=mult2.output, right=mult1.output)
        return model


class TestExtendExtractionImmediateValues(ComparisonTestBase):
    def model1(self):
        model = Model()
        model |= (add := Add())
        output = add.output + 2
        model |= Buffer().connect(output)
        return model

    def model2(self):
        model = Model()
        model |= (add1 := Add())
        model |= (add2 := Add(right=2)).connect(left=add1.output)
        model |= Buffer().connect(add2.output)
        return model


class TestExtendSingleFrozenSingleNonFrozenModel(ComparisonTestBase):
    def model1(self):
        model1 = Model()
        model1 |= (add1 := Add())
        model1._freeze()

        model2 = Model()
        model2 |= (add2 := Add())
        model2 |= Buffer().connect(add1.output * add2.output)

        return model2

    def model2(self):
        model1 = Model()
        model1 |= (_add1 := Add())
        model1._freeze()

        model2 = Model()
        model2 |= model1
        model2 |= (_add2 := Add())
        model2 |= (mult := Multiply()).connect(left=_add2.output, right=_add1.output)
        model2 |= Buffer().connect(mult.output)

        return model2


class TestExtendNonFrozenModelFrozenModel(ComparisonTestBase):
    def model1(self):
        out1 = IOKey("out1")
        out2 = IOKey("out2")

        model1 = Model()
        model1 |= Add().connect(output=out1)
        model2 = Model()
        model2 |= Add().connect(output=out2)
        model2._freeze()

        output = out1 + out2
        model1 |= Buffer().connect(output)
        return model1

    def model2(self):
        out1 = IOKey("out1")
        out2 = IOKey("out2")
        model1 = Model()
        model1 |= (add := Add()).connect(output=out1)
        model2 = Model()
        model2 |= Add().connect(output=out2)

        model1 |= model2
        model1 |= (add := Add()).connect(out1, out2)
        model1 |= Buffer().connect(add.output)

        return model1


class TestExtendCheckMetadata(ComparisonTestBase):
    def model1(self):
        weight_key = IOKey("weight")
        t_w = weight_key.transpose()
        m = Model()
        m |= Buffer().connect(t_w)
        assert list(m.dag.keys())[0].input.metadata == m.weight.metadata  # type: ignore

        model = Model()
        model |= m.connect(weight=IOKey("weight"))
        assert list(m.dag.keys())[0].input.metadata == m.weight.metadata  # type: ignore
        return model

    def model2(self):
        weight_key = IOKey("weight")
        m = Model()
        m |= Transpose().connect(weight_key)
        m += Buffer()
        assert list(m.dag.keys())[0].input.metadata == m.weight.metadata  # type: ignore

        model = Model()
        model |= m.connect(weight=IOKey("weight"))
        assert list(m.dag.keys())[0].input.metadata == m.weight.metadata  # type: ignore
        return model


class TestExtendProvisionalModel(ComparisonTestBase):
    def model1(self):
        model = Model()
        model |= Add().connect(left="left", right="right", output="output")
        return model

    def model2(self):
        model = Model()
        model |= Add().connect(left="left", right="right", output="output")
        model.output**2  # type: ignore
        return model


class TestExtendProvisionalModel2(ComparisonTestBase):
    def model1(self):
        model = Model()
        model |= Add().connect(left="left", right="right", output="output")
        pow = model.output**2  # type: ignore
        assert pow.model.provisional_source == model
        buf_model = Buffer()
        model |= buf_model.connect(pow)
        return model

    def model2(self):
        model = Model()
        model |= Add().connect(left="left", right="right", output="output")
        model |= Power().connect(model.output, 2)  # type: ignore
        model += Buffer()
        return model


class TestExtendConcat(ComparisonTestBase):
    def model1(self):
        model = Model()
        model |= (buff1 := Buffer())
        model |= (buff2 := Buffer())
        buff1_1d = buff1.output.atleast_1d()
        buff2_1d = buff2.output.atleast_1d()
        model |= Concat().connect(input=[buff1_1d, buff2_1d])
        return model

    def model2(self):
        model = Model()
        model |= (buff1 := Buffer())
        model |= (buff2 := Buffer())
        model |= (buff1_1d := AtLeast1D()).connect(buff1.output)
        model |= (buff2_1d := AtLeast1D()).connect(buff2.output)
        model |= (list_m := ToList(2)).connect(
            input1=buff1_1d.output, input2=buff2_1d.output
        )
        model |= Concat().connect(input=list_m.output)
        return model


class TestExtendOnlyDependentSubmodels(ComparisonTestBase):
    def model1(self):
        model = Model()
        model |= (buff1 := Buffer())
        a = buff1.output**2
        b = buff1.output / 3
        c = a + 4
        assert a.model is b.model is c.model and a.model is not None
        provisional_model = b.model
        assert provisional_model is not None
        dag = provisional_model.dag
        assert {m.__class__.__name__ for m in dag} == {"PowerOp", "AddOp", "DivideOp"}

        model |= Buffer().connect(c)

        assert b.model is provisional_model
        return model

    def model2(self):
        model = Model()
        model |= (buff := Buffer())
        model |= (pow := Power()).connect(buff.output, 2)
        model |= (add := Add()).connect(pow.output, 4)
        model |= Buffer().connect(add.output)
        return model


class TestExtendMergeWhileProvisionalModelCreated(ComparisonTestBase):
    def model1(self):
        model = Model()
        model |= (add := Add())
        a = add.output**2
        _ = add.output / 3
        c = a + 4
        model.merge_connections(add.left, add.right)
        model |= Buffer().connect(c)
        return model

    def model2(self):
        con = IOKey()
        model = Model()
        model |= (add := Add()).connect(con, con)
        model |= (pow := Power()).connect(add.output, 2)
        model |= (add := Add()).connect(pow.output, 4)
        model |= Buffer().connect(add.output)
        return model


class TestFunctionalModel(ComparisonTestBase):
    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        x = Add()(input1, input2)
        x = Multiply()(x, x)
        x = x**2  # type: ignore
        assert x.model is not None
        return x.model

    def model2(self):
        model = Model()
        model |= (add := Add()).connect("input1", "input2")
        model |= (mult := Multiply()).connect(add.output, add.output)
        model |= Power().connect(mult.output, 2)
        return model


class TestFunctionalModelWithLinear(ComparisonTestBase):
    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        x = Add()(left=input1, right=input2)
        x = Multiply()(left=x, right=x)
        x = x**2  # type: ignore
        x = Linear()(input=x)
        return x.model.parent  # type: ignore

    def model2(self):
        model = Model()
        model |= (add := Add()).connect("input1", "input2")
        model |= (mult := Multiply()).connect(add.output, add.output)
        model |= (pow := Power()).connect(mult.output, 2)
        model |= Linear().connect(input=pow.output)
        return model


class TestExposingExistingOutputConnection(ComparisonTestBase):
    def model1(self):
        input = IOKey("input")
        out = Buffer()(input=input)
        a = out**2
        return Model.create(output=a, buff_out=out)

    def model2(self):
        # TODO: Remove set_cout from second model after fixing the issue with couts.
        model = Model()
        model |= (buff := Buffer()).connect("input", "buff_out")
        model |= Power(exponent=2).connect(buff.output, output="output")
        model.expose_keys("output", "buff_out")
        model.set_cout("output", "buff_out")
        model._freeze()
        return model


class TestFunctionalModelWithCreateApi(ComparisonTestBase):
    skip_flat_graph_assertion = (True, "output is not named so compile raises error")

    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        x = Add()(left=input1, right=input2)
        x = Multiply()(left=x, right=x)
        x = x**2  # type: ignore
        x = Linear()(input=x)
        return Model.create(x)  # type: ignore

    def model2(self):
        model = Model()
        model |= (add := Add()).connect("input1", "input2")
        model |= (mult := Multiply()).connect(add.output, add.output)
        model |= (pow := Power(exponent=2)).connect(mult.output)
        model |= Linear().connect(input=pow.output)
        model._freeze()
        return model


class TestFunctionalModelWithCreateApiNoKeyNamings(ComparisonTestBase):
    skip_flat_graph_assertion = (True, "output is not named so compile raises error")

    def model1(self):
        input1 = IOKey()
        input2 = IOKey()
        x = Add()(left=input1, right=input2)
        x = Multiply()(left=x, right=x)
        x = x**2  # type: ignore
        x = Linear()(input=x)
        return Model.create(x)  # type: ignore

    def model2(self):
        model = Model()
        model |= (add := Add())
        model |= (mult := Multiply()).connect(add.output, add.output)
        model |= (pow := Power(exponent=2)).connect(mult.output)
        model |= Linear().connect(input=pow.output)
        model._freeze()
        return model


class TestFunctionalModelWithCreateApiWithImmediateInCall(ComparisonTestBase):
    def model1(self):
        input1 = IOKey()
        x = Add()(left=input1, right=3)
        x = Multiply()(left=x, right=x)
        x = x**2  # type: ignore
        x = Linear()(input=x)
        return Model.create(output=x)

    def model2(self):
        model = Model()
        model |= (add := Add(right=3)).connect()
        model |= (mult := Multiply()).connect(add.output, add.output)
        model |= (pow := Power(exponent=2)).connect(mult.output)
        model |= Linear().connect(input=pow.output, output=IOKey("output"))
        model._freeze()
        return model


class TestFunctionalModelWithCreateApiWithImmediateInModelInit(ComparisonTestBase):
    def model1(self):
        input1 = IOKey()
        x = Add(right=3)(left=input1)
        return Model.create(output=x)  # type: ignore

    def model2(self):
        model = Model()
        model |= Add(right=3).connect(output=IOKey("output"))
        model._freeze()
        return model


class TestFunctionalPartialModelCreation1(ComparisonTestBase):
    def model1(self):
        input = IOKey("input")
        t_input = Transpose(axes=(0, 2, 3, 1))(input=input)  # type: ignore
        _ = t_input**2
        b_out = Buffer()(input=t_input)
        return Model.create(output=b_out)

    def model2(self):
        input = IOKey("input")
        t_input = input.transpose((0, 2, 3, 1))
        _ = t_input**2
        model = Model()
        model |= Buffer().connect(input=t_input, output=IOKey("output"))
        model._freeze()
        return model


class TestFunctionalPartialModelCreation2(ComparisonTestBase):
    def model1(self):
        input = IOKey("input")
        t_input = Transpose(axes=(0, 2, 3, 1))(input=input)  # type: ignore
        _ = t_input**2
        b_out = Buffer()(input=t_input)
        return Model.create(output=b_out)

    def model2(self):
        input = IOKey("input")
        t_input = input.transpose((0, 2, 3, 1))
        _ = t_input**2
        b_out = Buffer()(input=t_input)
        return Model.create(output=b_out)


class TestFunctionalModelUnnamedInputkeys(ComparisonTestBase):
    def model1(self):
        x = IOKey("input", shape=[None, 512, None, None])
        normalized = GroupNorm(num_groups=32, eps=1e-6, name="norm")(x)
        return Model.create(normalized=normalized)

    def model2(self):
        x = IOKey("input", shape=[None, 512, None, None])
        model = Model()
        model |= GroupNorm(num_groups=32, eps=1e-6, name="norm").connect(
            x, "normalized"
        )
        model._freeze()
        return model


class TestFunctionalModelNaming(ComparisonTestBase):
    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        x = my_lin(input1, input2, name="my_lin")
        model = Model.create(output=x)
        assert list(model.dag)[0].name == "my_lin"
        return model

    def model2(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        x = my_lin(input1, input2)
        model = Model.create(output=x)
        return model


class TestFunctionalModelWithCallConcat(ComparisonTestBase):
    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        x = Concat()(input=[input1**2, Add(right=1)(input2)])
        return Model.create(output=x)

    def model2(self):
        model = Model()
        model |= (pow := Power(exponent=2)).connect("input1")
        model |= (add := Add(right=1)).connect("input2")
        model |= Concat().connect(
            input=[pow.output, add.output], output=IOKey("output")
        )
        model._freeze()
        return model


class TestFunctionalAttnBlock(ComparisonTestBase):
    n_channels = 512
    skip_flat_graph_assertion = (True, "discuss flat graph assertion in this test")

    def model1(self):
        return attn_block_functional(self.n_channels)

    def model2(self):
        model = attn_block_with_connect(self.n_channels)
        model._freeze()
        return model


class TestFunctionalEuclideanDistance(ComparisonTestBase):
    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        output = (input1**2 + input2**2) ** (1 / 2)
        model = Model.create(output=output, name="euc_distance")

        output = model(IOKey("input1"), IOKey("input2"))

        return Model.create(output=output)

    def model2(self):
        @functional
        def euc_distance(input1: IOKey, input2: IOKey):
            return (input1**2 + input2**2) ** (1 / 2)

        output = euc_distance(IOKey("input1"), IOKey("input2"))

        return Model.create(output=output)


class TestFunctionalEuclideanDistanceKeyworded(ComparisonTestBase):
    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        output = (input1**2 + input2**2) ** (1 / 2)
        model = Model.create(output=output, name="euc_distance")

        output = model(IOKey("input1"), IOKey("input2"))

        return Model.create(output=output)

    def model2(self):
        @functional
        def euc_distance(input1: IOKey, input2: IOKey):
            return (input1**2 + input2**2) ** (1 / 2)

        output = euc_distance(IOKey("input1"), input2=IOKey("input2"))

        return Model.create(output=output)


class TestFunctionalEuclideanDistanceNestedEvaluate(CompareEvaluateTestBase):
    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        output = (input1**2 + input2**2) ** (1 / 2)

        model1 = Model.create(output=output, name="euc_distance_1")
        model2 = deepcopy(model1)
        model2.name = "euc_distance_2"

        output1 = model1(input1=IOKey("input1"), input2=IOKey("input2"))
        output2 = model2(input1=IOKey("input3"), input2=IOKey("input4"))

        output = (output1**2 + output2**2) ** (1 / 2)

        model = Model.create(output=output, name="nested_euc_distance")

        output = model(
            input1=IOKey("input1", shape=[5, 10]),
            input2=IOKey("input2", shape=[5, 10]),
            input3=IOKey("input3", shape=[5, 10]),
            input4=IOKey("input4", shape=[5, 10]),
        )

        return Model.create(output=output)

    def model2(self):
        @functional
        def euc_distance(input1: IOKey, input2: IOKey):
            return (input1**2 + input2**2) ** (1 / 2)

        @functional
        def nested_euc_distance(
            input1: IOKey, input2: IOKey, input3: IOKey, input4: IOKey
        ):
            output1 = euc_distance(input1, input2)
            output2 = euc_distance(input3, input4)
            return (output1**2 + output2**2) ** (1 / 2)

        output = nested_euc_distance(
            IOKey("input1", shape=[5, 10]),
            IOKey("input2", shape=[5, 10]),
            IOKey("input3", shape=[5, 10]),
            IOKey("input4", shape=[5, 10]),
        )

        return Model.create(output=output)


class TestAttnBlockFunctionalVsDecorator(CompareEvaluateTestBase):
    n_channels: int = 512

    def model1(self):
        model = attn_block_functional(self.n_channels, name="attn-block_with_decorator")
        output = model(input=IOKey("input", shape=[2, 512, 32, 32]))
        new_model = Model.create(output=output, name="attn_block")
        return new_model

    def model2(self):
        input = IOKey("input", shape=[2, 512, 32, 32])
        output = attn_block_with_decorator(input, n_channels=self.n_channels)
        model = Model.create(output=output, name="attn_block")
        return model


class TestFunctionalWithTwoOutputs(CompareEvaluateTestBase):
    n_channels: int = 512

    def model1(self):
        input1 = IOKey("input1")
        input2 = IOKey("input2")

        output1 = ((input1 * input2) ** 2) * 7
        output2 = ((input1 - input2) + 3) ** 2

        model = Model.create(output1=output1, output2=output2)
        out1, out2 = model(IOKey("input1"), IOKey("input2"))  # type: ignore
        return Model.create(output1=out1, output2=out2)  # type: ignore

    def model2(self):
        @functional
        def model(input1: IOKey, input2: IOKey):
            output1 = ((input1 * input2) ** 2) * 7
            output2 = ((input1 - input2) + 3) ** 2
            return output1, output2

        out1, out2 = model(IOKey("input1"), input2=IOKey("input2"))
        return Model.create(output1=out1, output2=out2)


def test_extend_metadata_linear():
    lin1 = Linear()
    assert list(lin1.dag.keys())[0].input.metadata is lin1.weight.metadata  # type: ignore

    model = Model()
    model += lin1.connect(weight=IOKey("w"))
    assert list(lin1.dag.keys())[0].input.metadata is lin1.weight.metadata  # type: ignore
    assert lin1.weight.metadata is model.w.metadata  # type: ignore


def test_extend_canonicals_for_main_model():
    block = Model()
    buffer = Buffer()
    input = IOKey()
    block |= buffer.connect(input=input)
    assert block.cout.metadata == buffer.output.metadata
    input.sqrt()
    assert block.cout.metadata == buffer.output.metadata


def test_extend_canonicals_for_extract_model():
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    output = input1 * input2
    assert output.model is not None
    mult_model = list(output.model.dag.keys())[0]
    assert output.model.cout.metadata == mult_model.output.metadata  # type: ignore


def test_extend_error_shp_mismatch():
    input1 = IOKey("input1", shape=[3, 3])
    input2 = IOKey("input2", shape=[3, 2])
    with pytest.raises(ValueError) as err:
        input1 * input2

    assert (
        str(err.value)
        == "Inputs shape mismatch at dimension 1. Shapes are inconsistent."
    )


def test_extend_test_extend_multiple_non_frozen_models_error():
    model = Model()
    model |= (add := Add())

    model1 = Model()
    model1 |= (add2 := Add())

    with pytest.raises(ValueError) as err:
        add.output + add2.output
    assert str(err.value) == "Multiple non-frozen active models found in connections!"


def test_extend_test_extend_multiple_non_frozen_models_with_connection_error():
    out1 = IOKey("out1")
    out2 = IOKey("out2")

    model1 = Model()
    model1 |= Add().connect(output=out1)
    model2 = Model()
    model2 |= Add().connect(output=out2)

    with pytest.raises(ValueError) as err:
        out1 + out2
    assert str(err.value) == "Multiple non-frozen active models found in connections!"


def test_extend_error_by_constraint_solver():
    model = Model()
    buff = Buffer()
    model |= buff.connect(input="input1")
    model |= Add().connect(buff.output, IOKey(shape=[4, 4]))
    with pytest.raises(ValueError) as err:
        t = buff.input.T
        t.set_shapes([3, 4, 5])
    assert str(err.value) == "Possible values mismatch!"


def test_extend_error_by_constraint_solver_nested_model():
    model = Model()
    buff = Buffer()
    model |= buff.connect(input="input1")
    model |= Add().connect(buff.output, IOKey(shape=[4, 4]))
    parent_m = Model()
    parent_m |= model
    grand_parent_m = Model()
    grand_parent_m |= parent_m

    with pytest.raises(ValueError) as err:
        t = buff.input.T
        t.set_shapes([3, 4, 5])
    assert str(err.value) == "Possible values mismatch!"


def test_immediate_extend_integration():
    model = Model()
    query = IOKey("query", type=ml.Tensor)
    key = IOKey("key", type=ml.Tensor)
    bsz = query.transpose()

    q = query + 2
    model |= Buffer().connect(input=q)
    _key = key + 1
    k_r = _key + bsz
    model |= Buffer().connect(input=k_r)

    for con in model.conns.input_connections:
        assert con.model is model


def test_immediate_extend_integration_reshape():
    model = Model()
    queries = IOKey("queries")
    B = queries.shape[1]
    model |= Linear().connect(queries, output="in_proj")

    _ = model.in_proj.reshape((B, B, 3, -1))  # type: ignore

    for con in model.conns.input_connections:
        assert con.model is model


def test_immediate_extend_integration_str_matching():
    block = Model()
    input = IOKey("input")
    block += Buffer().connect(input="input", output="b_out")

    block |= Buffer().connect(input=input + block.b_out)  # type: ignore

    result = block.b_out + input  # type: ignore
    block |= Buffer().connect(result, output=IOKey("output"))

    for con in block.conns.input_connections:
        assert con.model is block


def test_immediate_extend_integration_str_matching2():
    block = Model()
    input = IOKey("input")
    block |= Buffer().connect(input="input", output="b_out")
    block |= Buffer().connect(input=input, output="b_odsfut")


def test_apply_rope_partial():
    block = Model()
    # We define the input connections
    xq = IOKey("xq", type=ml.Tensor)
    freqs_cis = IOKey("freqs_cis", type=ml.Tensor)

    xq_shape = xq.shape
    a, b, c, d = freqs_cis[..., 0], xq[..., 0], freqs_cis[..., 1], xq[..., 1]
    e = a * b
    f = c * d
    _ = e + f

    block |= Reshape().connect(shape=xq_shape, output="xq_out_raw")

    for con in block.conns.input_connections:
        assert con.model is block


def apply_rope(*, name: str | None = None) -> Model:
    block = Model(name=name)
    # We define the input connections
    xq = IOKey("xq", type=ml.Tensor)
    freqs_cis = IOKey("freqs_cis", type=ml.Tensor)

    xq_shape = xq.shape
    # Do the math
    a = (_a1 := freqs_cis[..., 0]) * (_a2 := xq[..., 0])
    b = freqs_cis[..., 1] * xq[..., 1]
    xq_out = a + b

    block |= Reshape().connect(xq_out, shape=xq_shape, output="xq_out_raw")
    return block


def test_apply_rope():
    block = apply_rope()
    for con in block.conns.input_connections:
        assert con.model is block


def build_attention_mask() -> Model:
    block = Model()
    block |= Arange(stop=77).connect(output="arange_out_1")
    block |= Arange(stop=77).connect(output="arange_out_2")
    upper_bool_triu = block.arange_out_1[..., None] >= block.arange_out_2[None, ...]  # type: ignore
    block |= Where().connect(
        cond=upper_bool_triu,
        input1=Tensor(0.0),
        input2=Tensor(float("-inf")),
        output=IOKey("output"),
    )
    return block


def test_multihead():
    d_model = 768
    n_head = 12
    block = Model()
    queries = IOKey("queries")
    head_dim = d_model // n_head
    B, L = queries.shape[0], queries.shape[1]
    block |= Linear(3 * d_model, name="in_proj").connect(queries, output="in_proj")

    in_proj = (
        block.in_proj.reshape((B, L, 3, -1))  # type: ignore
        .reshape((1, B, L, 3, d_model))
        .transpose((3, 1, 2, 0, 4))
        .reshape((3, B, L, -1))
    )

    queries = (
        in_proj[0, :, :, :].reshape((B, L, n_head, head_dim)).transpose((1, 2, 0, 3))
    )
    keys = in_proj[1, :, :, :].reshape((B, L, n_head, head_dim)).transpose((1, 2, 0, 3))
    values = (
        in_proj[2, :, :, :].reshape((B, L, n_head, head_dim)).transpose((1, 2, 0, 3))
    )

    block |= (mask_model := build_attention_mask())
    block |= ScaledDotProduct(is_causal=False, use_attn_mask=True).connect(
        query=queries,
        key=keys,
        value=values,
        attn_mask=mask_model.cout,
        output="attention",
    )
    _ = B * L
    for con in block.conns.input_connections:
        assert con.model is block


def test_rearrange():
    # This test is to check that the model can be extended with a provisional model's
    # input connection properly.
    block = Model()
    input = IOKey("input")
    input_shape = input.shape
    B, L = input_shape[0], input_shape[1]
    block |= Reshape().connect(input, shape=(B, L, 3, 10, -1))
    block += Transpose(axes=(2, 0, 3, 1, 4)).connect(output=IOKey("output"))
    block.expose_keys("output")
    for con in block.conns.input_connections:
        assert con.model is block


def test_extend_new_model_with_provisional_ref_count():
    submodel = Model()
    submodel |= (mult := Multiply())
    mult.output + 3
    sub_pro_model = submodel.provisional_model
    assert isinstance(sub_pro_model, Model)
    model = Model()
    model |= (buff := Buffer())
    buff.output - 2
    pro_model = model.provisional_model
    model |= submodel
    assert submodel.provisional_model is None
    assert sub_pro_model.provisional_source is False
    assert pro_model is model.provisional_model
    assert isinstance(model.provisional_model, Model)
    assert len(model.provisional_model.dag) == 2  # AddOp and SubtractOp
    assert sys.getrefcount(sub_pro_model.conns) == 4


def test_extend_new_model_with_provisional_model_connection_ref_count():
    submodel = Model()
    submodel |= (mult := Multiply())
    mult.output + 3
    sub_pro_model = submodel.provisional_model
    assert sys.getrefcount(sub_pro_model) == 7
    model = Model()
    model |= (buff := Buffer())
    buff.output - 2
    model |= submodel
    assert sys.getrefcount(sub_pro_model) == 2


def test_extend_child_provisional_extraction():
    submodel = Model()
    submodel |= (mult := Multiply())
    add_output = mult.output + 3

    model = Model()
    model |= (buff := Buffer())
    buff.output - 2
    model |= submodel
    pow = add_output**2
    model |= Buffer().connect(pow)
    for con in model.conns.input_connections:
        assert con.model is model


def test_flat_model_key_naming_matching():
    # This test is added to check that the key naming is consistent.
    # Cleaning conns objects was leading an error in the FlatModel
    # because of the metadata mismatch.
    submodel = Model()
    input = IOKey("input", type=Tensor)
    submodel |= Buffer().connect(output="arange")
    omega = submodel.arange + 2  # type: ignore
    out = input[..., None] * omega
    submodel |= Buffer().connect(input=out, output="dummy_out")

    model = Model()
    input = IOKey("input")

    model |= submodel.connect(input=input)

    metadata = list(list(model.dag.keys())[0].dag.keys())[2].input.metadata  # type: ignore
    assert model.conns.get_con_by_metadata(metadata) is not None  # type: ignore
    flat_model = FlatModel(
        model,
        ml.JaxBackend().op_function_dict,
        short_namings=False,
    )
    assert flat_model.queued_models == {}


def test_existing_cons_model_attribute_maps_to_model():
    input_key = IOKey("input")
    out = input_key + 2
    model = Model()
    model |= Buffer().connect(input=out)
    model |= Buffer().connect(input=input_key)
    assert input_key.model is not None


def attn_block_functional(n_channels: int, *, name: str | None = None):
    # Keep the original input for the residual connection.
    x = IOKey("input", shape=[None, 512, None, None])
    normalized = GroupNorm(num_groups=32, eps=1e-6, name="norm")(input=x)
    query = Convolution2D(1, n_channels, name="q")(input=normalized)
    key = Convolution2D(1, n_channels, name="k")(input=normalized)
    value = Convolution2D(1, n_channels, name="v")(input=normalized)
    shape = query.shape  # type: ignore

    query = query.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))  # type: ignore
    key = key.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))  # type: ignore
    value = value.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))  # type: ignore
    sdp_out = ScaledDotProduct(is_causal=False)(query=query, key=key, value=value)

    reshaped = Reshape()(input=sdp_out, shape=(shape[0], shape[2], shape[3], shape[1]))
    transposed = Transpose(axes=(0, 3, 1, 2))(input=reshaped)
    proj_out = Convolution2D(1, n_channels, name="proj_out")(input=transposed)

    return Model.create(output=proj_out + x, name=name)  # type: ignore


@functional
def attn_block_with_decorator(input: IOKey, *, n_channels: int):
    normalized = GroupNorm(num_groups=32, eps=1e-6, name="norm")(input=input)
    query = Convolution2D(1, n_channels, name="q")(input=normalized)
    key = Convolution2D(1, n_channels, name="k")(input=normalized)
    value = Convolution2D(1, n_channels, name="v")(input=normalized)
    shape = query.shape  # type: ignore

    query = query.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))  # type: ignore
    key = key.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))  # type: ignore
    value = value.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))  # type: ignore
    sdp_out = ScaledDotProduct(is_causal=False)(query=query, key=key, value=value)

    reshaped = Reshape()(input=sdp_out, shape=(shape[0], shape[2], shape[3], shape[1]))
    transposed = Transpose(axes=(0, 3, 1, 2))(input=reshaped)
    proj_out = Convolution2D(1, n_channels, name="proj_out")(input=transposed)
    return proj_out + input


def attn_block_with_connect(n_channels: int, *, name: str | None = None):
    block = Model(name=name)
    block |= GroupNorm(num_groups=32, eps=1e-6, name="norm").connect(
        IOKey("input", shape=[None, 512, None, None]), "normalized"
    )
    block |= Convolution2D(1, n_channels, name="q").connect(
        "normalized", output="query"
    )
    block |= Convolution2D(1, n_channels, name="k").connect("normalized", output="key")
    block |= Convolution2D(1, n_channels, name="v").connect(
        "normalized", output="value"
    )

    query = block.query  # type: ignore[attr-defined]
    key = block.key  # type: ignore[attr-defined]
    value = block.value  # type: ignore[attr-defined]

    shape = query.shape  # type: ignore[attr-defined]

    query = query.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    key = key.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    value = value.transpose((0, 2, 3, 1)).reshape((shape[0], 1, -1, shape[1]))
    block |= ScaledDotProduct(is_causal=False).connect(
        query, key, value, output="sdp_out"
    )
    block.set_cout("sdp_out")

    block += Reshape().connect(shape=(shape[0], shape[2], shape[3], shape[1]))
    block += Transpose(axes=(0, 3, 1, 2))
    block += Convolution2D(1, n_channels, name="proj_out")
    block += Add().connect(right="input", output=IOKey("output"))

    return block


@functional
def my_lin(left, right):
    scale = IOKey("scale")
    add_output = Add()(left=left, right=right)
    mult_out = Multiply()(left=left, right=add_output)
    return mult_out * scale  # type: ignore


def manual_functional_lin(left, right, name: str | None = None):
    _l, _r = IOKey(), IOKey()
    m = Model.create(my_lin(_l, _r), name=name)
    m.rename_key(_l, "left")
    m.rename_key(_r, "right")
    return m(left, right)


def test_functional_model_with_decorator():
    # Functional API with decorator
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = my_lin(input1, input2, name="my_lin")

    # Equivalent model using the |= operator

    # Create Square model
    model = Model(name="my_lin")
    model |= (add := Add()).connect(left="left", right="right")
    model |= (mult1 := Multiply()).connect("left", add.output)
    model |= Multiply().connect(mult1.output, "scale")
    # Create wrapper of lin_nested model
    parent_model = Model()
    parent_model |= model.connect(left="input1", right="input2")

    assert_models_equal(x.model.parent, parent_model)

    functional_model = Model.create(x)
    parent_model._freeze()
    assert_models_equal(functional_model, parent_model)

    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = manual_functional_lin(input1, input2, name="my_lin")
    manual_functional_model = Model.create(x)
    assert_models_equal(manual_functional_model, functional_model)


@functional
def square(arg):
    return arg**2


@functional
def lin_nested(left, right, *, scale_shp=None):
    scale = IOKey("scale", shape=scale_shp)
    add_output = Add()(left=left, right=right)
    mult_out = Multiply()(left=left, right=add_output)
    return mult_out * square(scale, name="my_square")


def test_functional_model_with_decorator_nested():
    # Functional API with nested functional model
    input1 = IOKey("input1")
    input2 = IOKey("input2")
    x = lin_nested(input1, input2, name="my_lin")
    functional_model = Model.create(x)

    # Equivalent model using the |= operator
    # Create Square model
    sq_model = Model("my_square")
    sq_model |= Square().connect("arg")

    # Create lin_nested model
    model = Model(name="my_lin")
    model |= (add := Add()).connect(left="left", right="right")
    model |= (mult1 := Multiply()).connect("left", add.output)
    model |= sq_model.connect(arg="scale")
    model |= Multiply().connect(mult1.output, sq_model.cout)

    # Create wrapper of lin_nested model
    parent_model = Model()
    parent_model |= model.connect(left="input1", right="input2")
    parent_model._freeze()

    assert_models_equal(functional_model, parent_model)


def test_model_create_output_order():
    for _ in range(20):
        input1 = IOKey("input1")
        input2 = IOKey("input2")
        x = Multiply()(input1, input2)
        y = Add()(input1, input2)
        z = Power()(input1, input2)
        model = Model.create(out1=x, out2=y, out3=z)
        assert model.output_keys == ["out1", "out2", "out3"]

        in1 = IOKey()
        in2 = IOKey()
        _out1, _out2, _out3 = model(in1, in2)  # type: ignore
        assert model.out1.metadata == _out1.metadata  # type: ignore
        assert model.out2.metadata == _out2.metadata  # type: ignore
        assert model.out3.metadata == _out3.metadata  # type: ignore


def check_constraints(model: Model):
    constraints = set()
    for key in model.conns.all.values():
        constraints |= key.metadata.all_constraints
    assert model.constraint_solver.constraint_map.keys() == constraints


def test_constraint_cleaning():
    hidden_states = IOKey("input")
    height = IOKey("height")

    height = Buffer()(input=height)
    hidden_states = Reshape()(input=hidden_states)

    model = Model.create(height_out=height, output=hidden_states)
    check_constraints(model)


def test_constraint_cleaning_unused_model():
    from mithril.models import Relu

    hidden_states = IOKey("input")
    height = IOKey("height")

    height = Buffer()(input=height)
    hidden_states = Relu()(input=hidden_states)
    _hidden_states = Relu()(input=hidden_states)
    _hidden_states = Reshape()(input=_hidden_states)

    model = Model.create(height_out=height, output=hidden_states)
    assert len(model.constraint_solver.constraint_map) == 3
    # The last Reshape model is not used and first Relu's connections do not contain
    # constraints of last Reshape model. Therefore, there will be only 3 constraints
    # in the constraint_map which are 1 buffer_constraint and 2 general_type_constraint
    # and the last Reshape model constraint.
    check_constraints(model)


def test_constraint_cleaning_lin():
    model = Linear()
    check_constraints(model)


def test_constraint_cleaning_manual_lin():
    weight_key = IOKey(name="weight", differentiable=True).transpose()

    model = Model()
    model |= Buffer().connect(weight_key)
    check_constraints(model)


def test_constraint_cleaning_composite():
    in1 = IOKey(name="input1").transpose()
    in2 = IOKey(name="input2").transpose()
    model = Model()
    model |= Add().connect(in1, in2)
    check_constraints(model)
