#include "bora/philote_discipline.h"

#include <array>
#include <stdexcept>
#include <string>

namespace bora {

namespace {

/// One named discipline input and its slice of the 23-entry gradient vector.
struct InputBlock {
    const char* name;
    size_t offset;
    size_t size;
};

constexpr std::array<InputBlock, 9> kInputBlocks{{
    {.name = "upper_weights",
     .offset = input_index::kUpperWeights,
     .size = kNumCstWeightsPerSide},
    {.name = "lower_weights",
     .offset = input_index::kLowerWeights,
     .size = kNumCstWeightsPerSide},
    {.name = "leading_edge_weight",
     .offset = input_index::kLeadingEdgeWeight,
     .size = 1},
    {.name = "TE_thickness", .offset = input_index::kTeThickness, .size = 1},
    {.name = "alpha", .offset = input_index::kAlpha, .size = 1},
    {.name = "Re", .offset = input_index::kRe, .size = 1},
    {.name = "n_crit", .offset = input_index::kNCrit, .size = 1},
    {.name = "xtr_upper", .offset = input_index::kXtrUpper, .size = 1},
    {.name = "xtr_lower", .offset = input_index::kXtrLower, .size = 1},
}};

constexpr std::array<const char*, 6> kScalarOutputs{
    "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr", "analysis_confidence",
};

constexpr std::array<const char*, 6> kBlOutputs{
    "upper_theta", "upper_H", "upper_ue_over_vinf",
    "lower_theta", "lower_H", "lower_ue_over_vinf",
};

ModelSize ModelSizeFromString(const std::string& name) {
    constexpr std::array<ModelSize, 8> kSizes{
        ModelSize::kXXSmall, ModelSize::kXSmall,   ModelSize::kSmall,
        ModelSize::kMedium,  ModelSize::kLarge,    ModelSize::kXLarge,
        ModelSize::kXXLarge, ModelSize::kXXXLarge,
    };
    for (const ModelSize size : kSizes) {
        if (ToString(size) == name) {
            return size;
        }
    }
    throw std::invalid_argument("Unknown model_size '" + name +
                                "' (expected one of xxsmall, xsmall, small, "
                                "medium, large, xlarge, xxlarge, xxxlarge)");
}

AeroInput ReadAeroInput(const philote::Variables& inputs) {
    AeroInput input;
    const philote::Variable& upper = inputs.at("upper_weights");
    const philote::Variable& lower = inputs.at("lower_weights");
    for (size_t i = 0; i < kNumCstWeightsPerSide; ++i) {
        input.kulfan.upper_weights[i] = upper(i);
        input.kulfan.lower_weights[i] = lower(i);
    }
    input.kulfan.leading_edge_weight = inputs.at("leading_edge_weight")(0);
    input.kulfan.TE_thickness = inputs.at("TE_thickness")(0);
    input.alpha = inputs.at("alpha")(0);
    input.Re = inputs.at("Re")(0);
    input.n_crit = inputs.at("n_crit")(0);
    input.xtr_upper = inputs.at("xtr_upper")(0);
    input.xtr_lower = inputs.at("xtr_lower")(0);
    return input;
}

/// Copies one scalar output's gradient into its per-input jacobian blocks.
void FillScalarPartials(philote::Partials& partials, const std::string& output,
                        const Gradient& gradient) {
    for (const InputBlock& block : kInputBlocks) {
        philote::Variable& jac = partials[{output, block.name}];
        for (size_t i = 0; i < block.size; ++i) {
            jac(i) = gradient[block.offset + i];
        }
    }
}

/// Copies a boundary layer distribution's gradients into its jacobian blocks.
/// Each block is row-major [kNumBlPoints x block.size].
void FillBlPartials(philote::Partials& partials, const std::string& output,
                    const std::array<Gradient, kNumBlPoints>& gradients) {
    for (const InputBlock& block : kInputBlocks) {
        philote::Variable& jac = partials[{output, block.name}];
        for (size_t row = 0; row < kNumBlPoints; ++row) {
            for (size_t i = 0; i < block.size; ++i) {
                jac(row * block.size + i) = gradients[row][block.offset + i];
            }
        }
    }
}

}  // namespace

void BoraDiscipline::Initialize() {
    AddOption("model_size", "string");
    AddOption("data_directory", "string");
    AddOption("include_boundary_layer", "bool");
}

void BoraDiscipline::SetOptions(const google::protobuf::Struct& options) {
    const auto& fields = options.fields();

    if (fields.count("model_size") &&
        fields.at("model_size").has_string_value()) {
        model_size_ =
            ModelSizeFromString(fields.at("model_size").string_value());
    }
    if (fields.count("data_directory") &&
        fields.at("data_directory").has_string_value()) {
        data_directory_ = fields.at("data_directory").string_value();
    }
    if (fields.count("include_boundary_layer")) {
        include_boundary_layer_ =
            fields.at("include_boundary_layer").bool_value();
    }

    // Rebuild the model eagerly so bad options fail the SetOptions RPC rather
    // than the first Compute.
    model_ = std::make_unique<NeuralFoil>(model_size_, data_directory_);

    // Invokes Configure() per the Philote option-handling contract.
    philote::ExplicitDiscipline::SetOptions(options);
}

const NeuralFoil& BoraDiscipline::Model() {
    if (!model_) {
        model_ = std::make_unique<NeuralFoil>(model_size_, data_directory_);
    }
    return *model_;
}

void BoraDiscipline::Setup() {
    const auto n_weights = static_cast<int64_t>(kNumCstWeightsPerSide);
    AddInput("upper_weights", {n_weights}, "");
    AddInput("lower_weights", {n_weights}, "");
    AddInput("leading_edge_weight", {1}, "");
    AddInput("TE_thickness", {1}, "");
    AddInput("alpha", {1}, "deg");
    AddInput("Re", {1}, "");
    AddInput("n_crit", {1}, "");
    AddInput("xtr_upper", {1}, "");
    AddInput("xtr_lower", {1}, "");

    for (const char* name : kScalarOutputs) {
        AddOutput(name, {1}, "");
    }
    if (include_boundary_layer_) {
        const auto n_points = static_cast<int64_t>(kNumBlPoints);
        for (const char* name : kBlOutputs) {
            AddOutput(name, {n_points}, "");
        }
    }
}

void BoraDiscipline::SetupPartials() {
    for (const char* output : kScalarOutputs) {
        for (const InputBlock& block : kInputBlocks) {
            DeclarePartials(output, block.name);
        }
    }
    if (include_boundary_layer_) {
        for (const char* output : kBlOutputs) {
            for (const InputBlock& block : kInputBlocks) {
                DeclarePartials(output, block.name);
            }
        }
    }
}

void BoraDiscipline::Compute(const philote::Variables& inputs,
                             philote::Variables& outputs) {
    const AeroOutput aero = Model().Analyze(ReadAeroInput(inputs));

    outputs.at("CL")(0) = aero.CL;
    outputs.at("CD")(0) = aero.CD;
    outputs.at("CM")(0) = aero.CM;
    outputs.at("Top_Xtr")(0) = aero.Top_Xtr;
    outputs.at("Bot_Xtr")(0) = aero.Bot_Xtr;
    outputs.at("analysis_confidence")(0) = aero.analysis_confidence;

    if (include_boundary_layer_) {
        for (size_t i = 0; i < kNumBlPoints; ++i) {
            outputs.at("upper_theta")(i) = aero.upper.theta[i];
            outputs.at("upper_H")(i) = aero.upper.H[i];
            outputs.at("upper_ue_over_vinf")(i) = aero.upper.ue_over_vinf[i];
            outputs.at("lower_theta")(i) = aero.lower.theta[i];
            outputs.at("lower_H")(i) = aero.lower.H[i];
            outputs.at("lower_ue_over_vinf")(i) = aero.lower.ue_over_vinf[i];
        }
    }
}

void BoraDiscipline::ComputePartials(const philote::Variables& inputs,
                                     philote::Partials& partials) {
    const AeroResult result =
        Model().AnalyzeWithGradients(ReadAeroInput(inputs));
    const AeroGradients& grad = result.gradient;

    FillScalarPartials(partials, "CL", grad.CL);
    FillScalarPartials(partials, "CD", grad.CD);
    FillScalarPartials(partials, "CM", grad.CM);
    FillScalarPartials(partials, "Top_Xtr", grad.Top_Xtr);
    FillScalarPartials(partials, "Bot_Xtr", grad.Bot_Xtr);
    FillScalarPartials(partials, "analysis_confidence",
                       grad.analysis_confidence);

    if (include_boundary_layer_) {
        FillBlPartials(partials, "upper_theta", grad.upper.theta);
        FillBlPartials(partials, "upper_H", grad.upper.H);
        FillBlPartials(partials, "upper_ue_over_vinf", grad.upper.ue_over_vinf);
        FillBlPartials(partials, "lower_theta", grad.lower.theta);
        FillBlPartials(partials, "lower_H", grad.lower.H);
        FillBlPartials(partials, "lower_ue_over_vinf", grad.lower.ue_over_vinf);
    }
}

}  // namespace bora
