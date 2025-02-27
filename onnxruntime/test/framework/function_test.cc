// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "onnx/defs/parser.h"

#include "core/common/span_utils.h"
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"

#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/common/tensor_op_test_utils.h"


// Unit tests to check the implementation of functions, model-local functions,
// function-inlining etc.

namespace onnxruntime {
namespace test {

static void Check(const char* source,
                  const char* input_name, std::vector<float> input_values,
                  const char* output_name, std::vector<float> output_values) {
  // Convert source-representation of model to ModelProto:
  ONNX_NAMESPACE::OnnxParser parser(source);
  ONNX_NAMESPACE::ModelProto model;
  auto parse_status = parser.Parse(model);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  // Serialize and then load model:
  std::string serialized_model;
  const bool serialization_status = model.SerializeToString(&serialized_model);
  ASSERT_TRUE(serialization_status) << "Failed to serialize proto to string";

  SessionOptions session_options;
  InferenceSession session_object{session_options, GetEnvironment()};

  std::stringstream sstr(serialized_model);
  auto status = session_object.Load(sstr);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options;
  run_options.run_tag = session_options.session_logid;

  NameMLValMap feeds;

  std::unique_ptr<CPUExecutionProvider> provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  OrtValue ort_value;
  CreateMLValue<float>(provider->GetAllocator(0, OrtMemTypeDefault), {int64_t(input_values.size())}, input_values, &ort_value);

  feeds.insert(std::make_pair(std::string(input_name), ort_value));

  std::vector<OrtValue> fetches;

  status = session_object.Run(run_options, feeds, AsSpan({std::string(output_name)}), &fetches);
  ASSERT_TRUE(status.IsOK()) << "Session Run failed: " << status.ErrorMessage() << std::endl;

  auto& tensor = fetches[0].Get<Tensor>();
  size_t size = static_cast<size_t>(tensor.Shape().Size());
  EXPECT_EQ(size, output_values.size());

  auto* data = tensor.Data<float>();
  float threshold = 0.001f;

  for (size_t i = 0; i < size; ++i) {
    ASSERT_NEAR(data[i], output_values[i], threshold) << "at position i:" << i;
  }
}

TEST(FunctionTest, Basic) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.myfun (x)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {2.0, 4.0, 6.0});
}

// Check that variables are renamed to avoid conflicts when multiple
// calls are inlined.
TEST(FunctionTest, Renaming) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y1 = local.myfun (x)
            y = local.myfun (y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {4.0, 8.0, 12.0});
}

// Check variable renaming in subgraphs.
// Scenario: input lx is used within subgraphs, but not in main graph.
// Both must be renamed to match the actual parameter name.
TEST(FunctionTest, InputInSubgraph) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            f = Constant <value = bool {0}> ()
            t = Constant <value = bool {1}> ()
            y1 = local.myfun (f, x)
            y = local.myfun (t, y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (b, lx) => (ly) {
            ly = If (b) <
                then_branch = g1 () => (float[N] z_then)
                {
                    two = Constant <value = float[1] {2.0}> ()
                    z_then =  Mul (lx, two)
                },
                else_branch = g2 () => (float[N] z_else)
                {
                    three = Constant <value = float[1] {3.0}> ()
                    z_else =  Mul (lx, three)
                }
                >
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {6.0, 12.0, 18.0});
}

// Check variable renaming in subgraphs.
// Scenario: intermediate temp is used within subgraphs, defined in main graph.
// Both must be renamed with a unique temporary name.
TEST(FunctionTest, TempInSubgraph) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            f = Constant <value = bool {0}> ()
            t = Constant <value = bool {1}> ()
            y1 = local.myfun (f, x)
            y = local.myfun (t, y1)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun (b, lx) => (ly) {
            temp = Identity (lx)
            ly = If (b) <
                then_branch = g1 () => (float[N] z_then)
                {
                    two = Constant <value = float[1] {2.0}> ()
                    z_then =  Mul (temp, two)
                },
                else_branch = g2 () => (float[N] z_else)
                {
                    three = Constant <value = float[1] {3.0}> ()
                    z_else =  Mul (temp, three)
                }
                >
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {6.0, 12.0, 18.0});
}

// Test a function body that calls another function.
TEST(FunctionTest, NestedCall) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.myfun (x)
        }

        <
        opset_import: [ "" : 16, "local" : 1],
        domain: "local"
        >
        myfun (lx) => (ly) {
            one = Constant <value = float[1] {1.0}> ()
            tmp = local.twice (lx)
            ly = Add (tmp, one)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        twice (lx) => (ly) {
            two = Constant <value = float[1] {2.0}> ()
            ly = Mul (lx, two)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {3.0, 5.0, 7.0});
}

// Nested call inside a conditional statement.
TEST(FunctionTest, CallInConditional) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            f = Constant <value = bool {0}> ()
            t = Constant <value = bool {1}> ()
            y1 = local.myfun (f, x)
            y = local.myfun (t, y1)
        }

        <
        opset_import: [ "" : 16, "local" : 1],
        domain: "local"
        >
        myfun (b, lx) => (ly) {
            temp = Identity (lx)
            ly = If (b) <
                then_branch = g1 () => (float[N] z_then)
                {
                    two = Constant <value = float[1] {2.0}> ()
                    z_then =  local.MulFun (temp, two)
                },
                else_branch = g2 () => (float[N] z_else)
                {
                    three = Constant <value = float[1] {3.0}> ()
                    z_else =  local.MulFun (temp, three)
                }
                >
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        MulFun (ax, bx) => (cx) {
            cx = Mul (ax, bx)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {6.0, 12.0, 18.0});
}

// Test use of attibute references, especially where source/target attribute
// names are not the same. In this example, the "start : int = @s" attribute-reference
// binds the attribute named "start" of the Shape op to the attribute named "s"
// of the containing function myfun.
TEST(FunctionTest, AttrName) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 16, "local" : 1 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            y = local.myfun <s = 0> (x)
        }

        <
        opset_import: [ "" : 16 ],
        domain: "local"
        >
        myfun <s> (lx) => (ly) {
            d = Shape <start : int = @s> (lx)
            df = Cast <to = 1> (d)
            ly = Mul (lx, df)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {3.0, 6.0, 9.0});
}

// Test use of constants inside sub-graphs, which are promoted to initializers by ORT.
TEST(FunctionTest, NestedConstant) {
  const char* code = R"(
        <
        ir_version: 8,
        opset_import: [ "" : 17 ]
        >
        agraph (float[N] x) => (float[N] y)
        {
            xseq = SequenceConstruct (x)
            yseq = SequenceMap (xseq) <body =
              zeropad (float[3] lx) => (float[6] ly) {
                zeros = Constant <value = float[3] {0.0, 0.0, 0.0}> ()
                ly = Concat <axis = 0> (lx, zeros)
              }>
            zero = Constant <value = int64{0}> ()
            y = SequenceAt (yseq, zero)
        }
        )";

  Check(code, "x", {1.0, 2.0, 3.0}, "y", {1.0, 2.0, 3.0, 0.0, 0.0, 0.0});
}

}  // namespace test
}  // namespace onnxruntime
