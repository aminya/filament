/*
 * Copyright (C) 2018 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <getopt/getopt.h>

#include <filaflat/BlobDictionary.h>
#include <filaflat/ChunkContainer.h>
#include <filaflat/DictionaryReader.h>
#include <filaflat/MaterialChunk.h>
#include <filaflat/ShaderBuilder.h>
#include <filaflat/Unflattener.h>

#include <filament/MaterialChunkType.h>
#include <filament/MaterialEnums.h>

#include <backend/DriverEnums.h>

#include <utils/Path.h>

#include <spirv_glsl.hpp>
#include <spirv-tools/libspirv.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>

using namespace filaflat;
using namespace filamat;
using namespace filament;
using namespace backend;
using namespace utils;

static const int alignment = 24;


class MaterialParser {
public:
    MaterialParser(Backend backend, const void* data, size_t size)
            : mChunkContainer(data, size), mMaterialChunk(mChunkContainer) {
        switch (backend) {
            case Backend::OPENGL:
                mMaterialTag = ChunkType::MaterialGlsl;
                mDictionaryTag = ChunkType::DictionaryGlsl;
                break;
            case Backend::METAL:
                mMaterialTag = ChunkType::MaterialMetal;
                mDictionaryTag = ChunkType::DictionaryMetal;
                break;
            case Backend::VULKAN:
                mMaterialTag = ChunkType::MaterialSpirv;
                mDictionaryTag = ChunkType::DictionarySpirv;
                break;
            default:
                break;
        }
    }

    bool parse() noexcept {
        if (mChunkContainer.parse()) {
            return mMaterialChunk.readIndex(mMaterialTag);
        }
        return false;
    }

    bool isShadingMaterial() const noexcept {
        ChunkContainer const& cc = mChunkContainer;
        return cc.hasChunk(MaterialName) && cc.hasChunk(MaterialVersion) &&
               cc.hasChunk(MaterialUib) && cc.hasChunk(MaterialSib) &&
               cc.hasChunk(MaterialShaderModels) && cc.hasChunk(mMaterialTag);
    }

    bool isPostProcessMaterial() const noexcept {
        ChunkContainer const& cc = mChunkContainer;
        return cc.hasChunk(PostProcessVersion)
               && cc.hasChunk(mMaterialTag) && cc.hasChunk(mDictionaryTag);
    }

    bool getShader(ShaderModel shaderModel,
            uint8_t variant, ShaderType stage, ShaderBuilder& shader) noexcept {

        ChunkContainer const& cc = mChunkContainer;
        if (!cc.hasChunk(mMaterialTag) || !cc.hasChunk(mDictionaryTag)) {
            return false;
        }

        BlobDictionary blobDictionary;
        if (!DictionaryReader::unflatten(cc, mDictionaryTag, blobDictionary)) {
            return false;
        }

        return mMaterialChunk.getShader(shader,
                blobDictionary, (uint8_t)shaderModel, variant, stage);
    }

private:
    ChunkContainer mChunkContainer;
    MaterialChunk mMaterialChunk;
    ChunkType mMaterialTag = ChunkType::Unknown;
    ChunkType mDictionaryTag = ChunkType::Unknown;
};

struct Config {
    bool printGLSL = false;
    bool printSPIRV = false;
    bool printMetal = false;
    bool transpile = false;
    bool binary = false;
    bool analyze = false;
    uint64_t shaderIndex;
};

struct ShaderInfo {
    ShaderModel shaderModel;
    uint8_t variant;
    ShaderType pipelineStage;
    uint32_t offset;
};

static void printUsage(const char* name) {
    std::string execName(utils::Path(name).getName());
    std::string usage(
            "MATINFO prints information about material files compiled with matc\n"
            "Usage:\n"
            "    MATINFO [options] <material file>\n"
            "\n"
            "Options:\n"
            "   --help, -h\n"
            "       Print this message\n\n"
            "   --print-glsl=[index], -g\n"
            "       Print GLSL for the nth shader (0 is the first OpenGL shader)\n\n"
            "   --print-spirv=[index], -s\n"
            "       Print disasm for the nth shader (0 is the first Vulkan shader)\n\n"
            "   --print-metal=[index], -m\n"
            "       Print Metal Shading Language for the nth shader (0 is the first Metal shader)\n\n"
            "   --print-vkglsl=[index], -v\n"
            "       Print the nth Vulkan shader transpiled into GLSL\n\n"
            "   --dump-binary=[index], -b\n"
            "       Dump binary SPIRV for the nth Vulkan shader to 'out.spv'\n\n"
            "   --license\n"
            "       Print copyright and license information\n\n"
            "   --analyze-spirv=[index], -a\n"
            "       Print annotated GLSL for the nth shader (0 is the first Vulkan shader)\n\n"
    );

    const std::string from("MATINFO");
    for (size_t pos = usage.find(from); pos != std::string::npos; pos = usage.find(from, pos)) {
        usage.replace(pos, from.length(), execName);
    }
    printf("%s", usage.c_str());
}

static void license() {
    std::cout <<
    #include "licenses/licenses.inc"
    ;
}

static int handleArguments(int argc, char* argv[], Config* config) {
    static constexpr const char* OPTSTR = "hlg:s:v:b:";
    static const struct option OPTIONS[] = {
            { "help",          no_argument,       0, 'h' },
            { "license",       no_argument,       0, 'l' },
            { "analyze-spirv", required_argument, 0, 'a' },
            { "print-glsl",    required_argument, 0, 'g' },
            { "print-spirv",   required_argument, 0, 's' },
            { "print-vkglsl",  required_argument, 0, 'v' },
            { "print-metal",   required_argument, 0, 'm' },
            { "dump-binary",   required_argument, 0, 'b' },
            { 0, 0, 0, 0 }  // termination of the option list
    };

    int opt;
    int optionIndex = 0;

    while ((opt = getopt_long(argc, argv, OPTSTR, OPTIONS, &optionIndex)) >= 0) {
        std::string arg(optarg ? optarg : "");
        switch (opt) {
            default:
            case 'h':
                printUsage(argv[0]);
                exit(0);
            case 'l':
                license();
                exit(0);
            case 'g':
                config->printGLSL = true;
                config->shaderIndex = static_cast<uint64_t>(std::stoi(arg));
                break;
            case 's':
                config->printSPIRV = true;
                config->shaderIndex = static_cast<uint64_t>(std::stoi(arg));
                break;
            case 'v':
                config->printSPIRV = true;
                config->shaderIndex = static_cast<uint64_t>(std::stoi(arg));
                config->transpile = true;
                break;
            case 'a':
                config->printSPIRV = true;
                config->shaderIndex = static_cast<uint64_t>(std::stoi(arg));
                config->analyze = true;
                break;
            case 'b':
                config->printSPIRV = true;
                config->shaderIndex = static_cast<uint64_t>(std::stoi(arg));
                config->binary = true;
                break;
            case 'm':
                config->printMetal = true;
                config->shaderIndex = static_cast<uint64_t>(std::stoi(arg));
        }
    }

    return optind;
}

static std::ifstream::pos_type getFileSize(const char* filename) {
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

template<typename T>
static bool read(const ChunkContainer& container, filamat::ChunkType type, T* value) noexcept {
    if (!container.hasChunk(type)) {
        return false;
    }

    Unflattener unflattener(container.getChunkStart(type), container.getChunkEnd(type));
    return unflattener.read(value);
}

template<typename T>
static const char* toString(T value);

template<>
const char* toString(Shading shadingModel) {
    switch (shadingModel) {
        case Shading::UNLIT: return "unlit";
        case Shading::LIT: return "lit";
        case Shading::SUBSURFACE: return "subsurface";
        case Shading::CLOTH: return "cloth";
        case Shading::SPECULAR_GLOSSINESS: return "specularGlossiness";
    }
}

template<>
const char* toString(BlendingMode blendingMode) {
    switch (blendingMode) {
        case BlendingMode::OPAQUE: return "opaque";
        case BlendingMode::TRANSPARENT: return "transparent";
        case BlendingMode::ADD: return "add";
        case BlendingMode::MASKED: return "masked";
        case BlendingMode::FADE: return "fade";
        case BlendingMode::MULTIPLY: return "multiply";
        case BlendingMode::SCREEN: return "screen";
    }
}

template<>
const char* toString(Interpolation interpolation) {
    switch (interpolation) {
        case Interpolation::SMOOTH: return "smooth";
        case Interpolation::FLAT: return "flat";
    }
}

template<>
const char* toString(VertexDomain domain) {
    switch (domain) {
        case VertexDomain::OBJECT: return "object";
        case VertexDomain::WORLD: return "world";
        case VertexDomain::VIEW: return "view";
        case VertexDomain::DEVICE: return "device";
    }
}

template<>
const char* toString(CullingMode cullingMode) {
    switch (cullingMode) {
        case CullingMode::NONE: return "none";
        case CullingMode::FRONT: return "front";
        case CullingMode::BACK: return "back";
        case CullingMode::FRONT_AND_BACK: return "front & back";
    }
}

template<>
const char* toString(TransparencyMode transparencyMode) {
    switch (transparencyMode) {
        case TransparencyMode::DEFAULT: return "default";
        case TransparencyMode::TWO_PASSES_ONE_SIDE: return "two passes, one side";
        case TransparencyMode::TWO_PASSES_TWO_SIDES: return "two passes, two sides";
    }
}

template<>
const char* toString(VertexAttribute attribute) {
    switch (attribute) {
        case VertexAttribute::POSITION: return "position";
        case VertexAttribute::TANGENTS: return "tangents";
        case VertexAttribute::COLOR: return "color";
        case VertexAttribute::UV0: return "uv0";
        case VertexAttribute::UV1: return "uv1";
        case VertexAttribute::BONE_INDICES: return "bone indices";
        case VertexAttribute::BONE_WEIGHTS: return "bone weights";
        case VertexAttribute::CUSTOM0: return "custom0";
        case VertexAttribute::CUSTOM1: return "custom1";
        case VertexAttribute::CUSTOM2: return "custom2";
        case VertexAttribute::CUSTOM3: return "custom3";
        case VertexAttribute::CUSTOM4: return "custom4";
        case VertexAttribute::CUSTOM5: return "custom5";
        case VertexAttribute::CUSTOM6: return "custom6";
        case VertexAttribute::CUSTOM7: return "custom7";
    }
    return "--";
}

template<>
const char* toString(bool value) {
    return value ? "true" : "false";
}

template<>
const char* toString(ShaderType stage) {
    switch (stage) {
        case ShaderType::VERTEX: return "vs";
        case ShaderType::FRAGMENT: return "fs";
        default: break;
    }
    return "--";
}

template<>
const char* toString(ShaderModel model) {
    switch (model) {
        case ShaderModel::UNKNOWN: return "--";
        case ShaderModel::GL_ES_30: return "gles30";
        case ShaderModel::GL_CORE_41: return "gl41";
    }
}

template<>
const char* toString(UniformType type) {
    switch (type) {
        case UniformType::BOOL:   return "bool";
        case UniformType::BOOL2:  return "bool2";
        case UniformType::BOOL3:  return "bool3";
        case UniformType::BOOL4:  return "bool4";
        case UniformType::FLOAT:  return "float";
        case UniformType::FLOAT2: return "float2";
        case UniformType::FLOAT3: return "float3";
        case UniformType::FLOAT4: return "float4";
        case UniformType::INT:    return "int";
        case UniformType::INT2:   return "int2";
        case UniformType::INT3:   return "int3";
        case UniformType::INT4:   return "int4";
        case UniformType::UINT:   return "uint";
        case UniformType::UINT2:  return "uint2";
        case UniformType::UINT3:  return "uint3";
        case UniformType::UINT4:  return "uint4";
        case UniformType::MAT3:   return "float3x3";
        case UniformType::MAT4:   return "float4x4";
    }
}

template<>
const char* toString(SamplerType type) {
    switch (type) {
        case SamplerType::SAMPLER_2D: return "sampler2D";
        case SamplerType::SAMPLER_CUBEMAP: return "samplerCubemap";
        case SamplerType::SAMPLER_EXTERNAL: return "samplerExternal";
    }
}

template<>
const char* toString(Precision precision) {
    switch (precision) {
        case Precision::LOW: return "lowp";
        case Precision::MEDIUM: return "mediump";
        case Precision::HIGH: return "highp";
        case Precision::DEFAULT: return "default";
    }
}

template<>
const char* toString(SamplerFormat format) {
    switch (format) {
        case SamplerFormat::INT: return "int";
        case SamplerFormat::UINT: return "uint";
        case SamplerFormat::FLOAT: return "float";
        case SamplerFormat::SHADOW: return "shadow";
    }
}

static std::string arraySizeToString(uint64_t size) {
    if (size > 1) {
        std::string s = "[";
        s += size;
        s += "]";
        return s;
    }
    return "";
}

template<typename T, typename V>
static void printChunk(const ChunkContainer& container, filamat::ChunkType type, const char* title) {
    T value;
    if (read(container, type, reinterpret_cast<V*>(&value))) {
        std::cout << "    " << std::setw(alignment) << std::left << title;
        std::cout << toString(value) << std::endl;
    }
}

static void printFloatChunk(const ChunkContainer& container, filamat::ChunkType type,
        const char* title) {
    float value;
    if (read(container, type, &value)) {
        std::cout << "    " << std::setw(alignment) << std::left << title;
        std::cout << std::setprecision(2) << value << std::endl;
    }
}

static void printUint32Chunk(const ChunkContainer& container, filamat::ChunkType type,
        const char* title) {
    uint32_t value;
    if (read(container, type, &value)) {
        std::cout << "    " << std::setw(alignment) << std::left << title;
        std::cout << value << std::endl;
    }
}

static bool printMaterial(const ChunkContainer& container) {
    std::cout << "Material:" << std::endl;

    uint32_t version;
    if (read(container, filamat::MaterialVersion, &version)) {
        std::cout << "    " << std::setw(alignment) << std::left << "Version: ";
        std::cout << version << std::endl;
    }

    printUint32Chunk(container, filamat::PostProcessVersion, "Post process version: ");

    CString name;
    if (read(container, filamat::MaterialName, &name)) {
        std::cout << "    " << std::setw(alignment) << std::left << "Name: ";
        std::cout << name.c_str() << std::endl;
    }

    std::cout << std::endl;

    std::cout << "Shading:" << std::endl;
    printChunk<Shading, uint8_t>(container, filamat::MaterialShading, "Model: ");
    printChunk<VertexDomain, uint8_t>(container, filamat::MaterialVertexDomain,
            "Vertex domain: ");
    printChunk<Interpolation, uint8_t>(container, filamat::MaterialInterpolation,
            "Interpolation: ");
    printChunk<bool, bool>(container, filamat::MaterialShadowMultiplier, "Shadow multiply: ");
    printChunk<bool, bool>(container, filamat::MaterialSpecularAntiAliasing, "Specular anti-aliasing: ");
    printFloatChunk(container, filamat::MaterialSpecularAntiAliasingVariance, "    Variance: ");
    printFloatChunk(container, filamat::MaterialSpecularAntiAliasingThreshold, "    Threshold: ");
    printChunk<bool, bool>(container, filamat::MaterialClearCoatIorChange, "Clear coat IOR change: ");

    std::cout << std::endl;

    std::cout << "Raster state:" << std::endl;
    printChunk<BlendingMode, uint8_t>(container, filamat::MaterialBlendingMode, "Blending: ");
    printFloatChunk(container, filamat::MaterialMaskThreshold, "Mask threshold: ");
    printChunk<bool, bool>(container, filamat::MaterialColorWrite, "Color write: ");
    printChunk<bool, bool>(container, filamat::MaterialDepthWrite, "Depth write: ");
    printChunk<bool, bool>(container, filamat::MaterialDepthTest, "Depth test: ");
    printChunk<bool, bool>(container, filamat::MaterialDoubleSided, "Double sided: ");
    printChunk<CullingMode, uint8_t>(container, filamat::MaterialCullingMode, "Culling: ");
    printChunk<TransparencyMode, uint8_t>(container, filamat::MaterialTransparencyMode, "Transparency: ");

    std::cout << std::endl;

    uint32_t requiredAttributes;
    if (read(container, filamat::MaterialRequiredAttributes, &requiredAttributes)) {
        AttributeBitset bitset;
        bitset.setValue(requiredAttributes);

        if (bitset.count() > 0) {
            std::cout << "Required attributes:" << std::endl;
            for (size_t i = 0; i < bitset.size(); i++) {
                if (bitset.test(i)) {
                    std::cout << "    " <<
                              toString(static_cast<VertexAttribute>(i)) << std::endl;
                }
            }
            std::cout << std::endl;
        }
    }

    return true;
}

static bool printParametersInfo(ChunkContainer container) {
    if (!container.hasChunk(filamat::ChunkType::MaterialUib)) {
        return true;
    }

    Unflattener uib(
            container.getChunkStart(filamat::ChunkType::MaterialUib),
            container.getChunkEnd(filamat::ChunkType::MaterialUib));

    CString name;
    if (!uib.read(&name)) {
        return false;
    }

    uint64_t uibCount;
    if (!uib.read(&uibCount)) {
        return false;
    }

    Unflattener sib(
            container.getChunkStart(filamat::ChunkType::MaterialSib),
            container.getChunkEnd(filamat::ChunkType::MaterialSib));

    if (!sib.read(&name)) {
        return false;
    }

    uint64_t sibCount;
    if (!sib.read(&sibCount)) {
        return false;
    }

    if (uibCount == 0 && sibCount == 0) {
        return true;
    }

    std::cout << "Parameters:" << std::endl;

    for (uint64_t i = 0; i < uibCount; i++) {
        CString fieldName;
        uint64_t fieldSize;
        uint8_t fieldType;
        uint8_t fieldPrecision;

        if (!uib.read(&fieldName)) {
            return false;
        }

        if (!uib.read(&fieldSize)) {
            return false;
        }

        if (!uib.read(&fieldType)) {
            return false;
        }

        if (!uib.read(&fieldPrecision)) {
            return false;
        }

        std::cout << "    "
                  << std::setw(alignment) << fieldName.c_str()
                  << std::setw(alignment) << toString(UniformType(fieldType))
                  << arraySizeToString(fieldSize)
                  << std::setw(10) << toString(Precision(fieldPrecision))
                  << std::endl;
    }

    for (uint64_t i = 0; i < sibCount; i++) {
        CString fieldName;
        uint8_t fieldType;
        uint8_t fieldFormat;
        uint8_t fieldPrecision;
        bool fieldMultisample;

        if (!sib.read(&fieldName)) {
            return false;
        }

        if (!sib.read(&fieldType)) {
            return false;
        }

        if (!sib.read(&fieldFormat))
            return false;

        if (!sib.read(&fieldPrecision)) {
            return false;
        }

        if (!sib.read(&fieldMultisample)) {
            return false;
        }

        std::cout << "    "
                << std::setw(alignment) << fieldName.c_str()
                << std::setw(alignment) << toString(SamplerType(fieldType))
                << std::setw(10) << toString(Precision(fieldPrecision))
                << toString(SamplerFormat(fieldFormat))
                << std::endl;
    }

    std::cout << std::endl;

    return true;
}

// Unpack a 64 bit integer into a std::string
inline utils::CString typeToString(uint64_t v) {
    uint8_t* raw = (uint8_t*) &v;
    char str[9];
    for (size_t i = 0; i < 8; i++) {
        str[7 - i] = raw[i];
    }
    str[8] = '\0';
    return utils::CString(str, strnlen(str, 8));
}

static void printChunks(const ChunkContainer& container) {
    std::cout << "Chunks:" << std::endl;

    std::cout << "    " << std::setw(9) << std::left << "Name ";
    std::cout << std::setw(7) << std::right << "Size" << std::endl;

    size_t count = container.getChunkCount();
    for (size_t i = 0; i < count; i++) {
        auto chunk = container.getChunk(i);
        std::cout << "    " << typeToString(chunk.type).c_str() << " ";
        std::cout << std::setw(7) << std::right << chunk.desc.size << std::endl;
    }
}

static bool getMetalShaderInfo(ChunkContainer container, std::vector<ShaderInfo>* info) {
    if (!container.hasChunk(filamat::ChunkType::MaterialMetal)) {
        return true; // that's not an error, a material can have no metal stuff
    }

    Unflattener unflattener(
            container.getChunkStart(filamat::ChunkType::MaterialMetal),
            container.getChunkEnd(filamat::ChunkType::MaterialMetal));

    uint64_t shaderCount = 0;
    if (!unflattener.read(&shaderCount) || shaderCount == 0) {
        return false;
    }

    info->clear();
    info->reserve(shaderCount);

    for (uint64_t i = 0; i < shaderCount; i++) {
        uint8_t shaderModelValue;
        uint8_t variantValue;
        uint8_t pipelineStageValue;
        uint32_t offsetValue;

        if (!unflattener.read(&shaderModelValue)) {
            return false;
        }

        if (!unflattener.read(&variantValue)) {
            return false;
        }

        if (!unflattener.read(&pipelineStageValue)) {
            return false;
        }

        if (!unflattener.read(&offsetValue)) {
            return false;
        }

        info->push_back({
                .shaderModel = ShaderModel(shaderModelValue),
                .variant = variantValue,
                .pipelineStage = ShaderType(pipelineStageValue),
                .offset = offsetValue
        });
    }

    return true;
}

static bool getGlShaderInfo(ChunkContainer container, std::vector<ShaderInfo>* info) {
    if (!container.hasChunk(filamat::ChunkType::MaterialGlsl)) {
        return true; // that's not an error, a material can have no glsl stuff
    }

    Unflattener unflattener(
            container.getChunkStart(filamat::ChunkType::MaterialGlsl),
            container.getChunkEnd(filamat::ChunkType::MaterialGlsl));

    uint64_t shaderCount;
    if (!unflattener.read(&shaderCount) || shaderCount == 0) {
        return false;
    }

    info->clear();
    info->reserve(shaderCount);

    for (uint64_t i = 0; i < shaderCount; i++) {
        uint8_t shaderModelValue;
        uint8_t variantValue;
        uint8_t pipelineStageValue;
        uint32_t offsetValue;

        if (!unflattener.read(&shaderModelValue)) {
            return false;
        }

        if (!unflattener.read(&variantValue)) {
            return false;
        }

        if (!unflattener.read(&pipelineStageValue)) {
            return false;
        }

        if (!unflattener.read(&offsetValue)) {
            return false;
        }

        info->push_back({
            .shaderModel = ShaderModel(shaderModelValue),
            .variant = variantValue,
            .pipelineStage = ShaderType(pipelineStageValue),
            .offset = offsetValue
        });
    }
    return true;
}

static bool getVkShaderInfo(ChunkContainer container, std::vector<ShaderInfo>* info) {
    if (!container.hasChunk(filamat::ChunkType::MaterialSpirv)) {
        return true; // that's not an error, a material can have no spirv stuff
    }

    Unflattener unflattener(
            container.getChunkStart(filamat::ChunkType::MaterialSpirv),
            container.getChunkEnd(filamat::ChunkType::MaterialSpirv));

    uint64_t shaderCount;
    if (!unflattener.read(&shaderCount) || shaderCount == 0) {
        return false;
    }

    info->clear();
    info->reserve(shaderCount);

    for (uint64_t i = 0; i < shaderCount; i++) {
        uint8_t shaderModelValue;
        uint8_t variantValue;
        uint8_t pipelineStageValue;
        uint32_t dictionaryIndex;

        if (!unflattener.read(&shaderModelValue)) {
            return false;
        }

        if (!unflattener.read(&variantValue)) {
            return false;
        }

        if (!unflattener.read(&pipelineStageValue)) {
            return false;
        }

        if (!unflattener.read(&dictionaryIndex)) {
            return false;
        }

        info->push_back({
            .shaderModel = ShaderModel(shaderModelValue),
            .variant = variantValue,
            .pipelineStage = ShaderType(pipelineStageValue),
            .offset = dictionaryIndex
        });
    }
    return true;
}

static void printShaderInfo(const std::vector<ShaderInfo>& info) {
    for (uint64_t i = 0; i < info.size(); ++i) {
        const auto& item = info[i];
        std::cout << "    #";
        std::cout << std::setw(4) << std::left << i;
        std::cout << std::setw(6) << std::left << toString(item.shaderModel);
        std::cout << " ";
        std::cout << std::setw(2) << std::left << toString(item.pipelineStage);
        std::cout << " ";
        std::cout << "0x" << std::hex << std::setfill('0') << std::setw(2)
                  << std::right << (int) item.variant;
        std::cout << std::setfill(' ') << std::dec << std::endl;
    }
    std::cout << std::endl;
}

static bool printGlslInfo(ChunkContainer container) {
    std::vector<ShaderInfo> info;
    if (!getGlShaderInfo(container, &info)) {
        return false;
    }
    std::cout << "GLSL shaders:" << std::endl;
    printShaderInfo(info);
    return true;
}

static bool printVkInfo(ChunkContainer container) {
    std::vector<ShaderInfo> info;
    if (!getVkShaderInfo(container, &info)) {
        return false;
    }
    std::cout << "Vulkan shaders:" << std::endl;
    printShaderInfo(info);
    return true;
}

static bool printMetalInfo(ChunkContainer container) {
    std::vector<ShaderInfo> info;
    if (!getMetalShaderInfo(container, &info)) {
        return false;
    }
    std::cout << "Metal shaders:" << std::endl;
    printShaderInfo(info);
    return true;
}

static bool printMaterialInfo(const ChunkContainer& container) {
    if (!printMaterial(container)) {
        return false;
    }

    if (!printParametersInfo(container)) {
        return false;
    }

    if (!printGlslInfo(container)) {
        return false;
    }

    if (!printVkInfo(container)) {
        return false;
    }

    if (!printMetalInfo(container)) {
        return false;
    }

    printChunks(container);

    std::cout << std::endl;

    return true;
}

// Consumes SPIRV binary and produces a GLSL-ES string.
static void transpileSpirv(const std::vector<uint32_t>& spirv) {
    using namespace spirv_cross;

    CompilerGLSL::Options emitOptions;
    emitOptions.es = true;
    emitOptions.vulkan_semantics = true;

    CompilerGLSL glslCompiler(move(spirv));
    glslCompiler.set_common_options(emitOptions);
    std::cout << glslCompiler.compile();
}

// Consumes SPIRV binary and produces an ordered map from "line number" to "GLSL string" where
// the line number is determined by line directives, and the GLSL string is one or more lines of
// transpiled GLSL-ES.
static std::map<int, std::string> transpileSpirvToLines(const std::vector<uint32_t>& spirv) {
    using namespace spirv_cross;

    CompilerGLSL::Options emitOptions;
    emitOptions.es = true;
    emitOptions.vulkan_semantics = true;
    emitOptions.emit_line_directives = true;

    CompilerGLSL glslCompiler(move(spirv));
    glslCompiler.set_common_options(emitOptions);
    std::string transpiled = glslCompiler.compile();

    std::map<int, std::string> result;
    const std::regex lineDirectivePattern("\\#line ([0-9]+)");

    std::istringstream ss(glslCompiler.compile());
    std::string glslCodeline;
    int currentLineNumber = -1;
    while (std::getline(ss, glslCodeline, '\n')) {
        std::smatch matchResult;
        if (std::regex_search(glslCodeline, matchResult, lineDirectivePattern)) {
            currentLineNumber = stoi(matchResult[1].str());
        } else {
            result[currentLineNumber] += glslCodeline + "\n";
        }
    }

    return result;
}

static void analyzeSpirv(const std::vector<uint32_t>& spirv, const char* disassembly) {
    using namespace std;

    const map<int, string> glsl = transpileSpirvToLines(spirv);
    const regex globalDecoratorPattern("OpDecorate (\\%[A-Za-z_0-9]+) RelaxedPrecision");
    const regex typeDefinitionPattern("(\\%[A-Za-z_0-9]+) = OpType[A-Z][a-z]+");
    const regex memberDecoratorPattern("OpMemberDecorate (\\%[A-Za-z_0-9]+) ([0-9]+) RelaxedPrecision");
    const regex lineDirectivePattern("OpLine (\\%[A-Za-z_0-9]+) ([0-9]+)");
    const regex binaryFunctionPattern("(\\%[A-Za-z_0-9]+).*(\\%[A-Za-z_0-9]+)");
    const regex operandPattern("(\\%[A-Za-z_0-9]+)");
    const regex operatorPattern("Op[A-Z][A-Za-z]+");
    const set<string> ignoredOperators = { "OpStore", "OpLoad", "OpAccessChain" };

    string spirvInstruction;
    smatch matchResult;

    // In the first pass, collect types and variables that are decorated as "relaxed".
    //
    // NOTE: We also collect struct members but do not use them in any analysis (yet).
    // In SPIR-V, struct fields are accessed through pointers, so we would need to:
    //
    // 1) Create a map of all OpConstant values (integers only, %int and %uint)
    // 2) Parse all "OpAccessChain" instructions and dereference their OpConstant arguments
    // 3) Follow through to the downstream OpStore / OpLoad
    //
    // Regarding struct field precision, the SPIR-V specification says:
    //
    //   When applied to a variable or structure member, all loads and stores from the decorated
    //   object may be treated as though they were decorated with RelaxedPrecision. Loads may also
    //   be decorated with RelaxedPrecision, in which case they are treated as operating at
    //   relaxed precision.

    set<string> relaxedPrecisionVariables;
    set<string> typeIds;
    {
        istringstream ss(disassembly);
        while (getline(ss, spirvInstruction, '\n')) {
            if (regex_search(spirvInstruction, matchResult, typeDefinitionPattern)) {
                typeIds.insert(matchResult[1].str());
            } else if (regex_search(spirvInstruction, matchResult, globalDecoratorPattern)) {
                relaxedPrecisionVariables.insert(matchResult[1].str());
            } else if (regex_search(spirvInstruction, matchResult, memberDecoratorPattern)) {
                string member = matchResult[1].str() + "." + matchResult[2].str();
                relaxedPrecisionVariables.insert(member);
            }
        }
    }

    // In the second pass, track line numbers and detect potential mixed precision.
    // The line directive are not necessarily in order (!) so we also track their ordering.

    map<int, string> mixedPrecisionInfo;
    vector<int> lineNumbers = {-1};
    {
        istringstream ss(disassembly);
        int currentLineNumber = -1;
        while (getline(ss, spirvInstruction, '\n')) {
            if (regex_search(spirvInstruction, matchResult, lineDirectivePattern)) {
                currentLineNumber = stoi(matchResult[2].str());
                lineNumbers.push_back(currentLineNumber);
            } else if (regex_search(spirvInstruction, matchResult, binaryFunctionPattern)) {

                // Trim out the leftmost whitespace.
                const string trimmed = regex_replace(spirvInstruction, regex("^\\s+"), string(""));

                // Ignore certain operators.
                regex_search(trimmed, matchResult, operatorPattern);
                if (ignoredOperators.count(matchResult[0]) || currentLineNumber == -1) {
                    continue;
                }

                // Check for mixed precision.
                bool mixed = false;
                int relaxed = -1;
                string remaining = trimmed;
                string info = trimmed + "; relaxed = ";
                while (regex_search(remaining, matchResult, operandPattern)) {
                    const string arg = matchResult[1].str();
                    if (typeIds.count(arg) == 0) {
                        if (relaxedPrecisionVariables.count(arg) > 0) {
                            mixed = relaxed == 0 ? true : mixed;
                            relaxed = 1;
                            info += arg + " ";
                        } else {
                            mixed = relaxed == 1 ? true : mixed;
                            relaxed = 0;
                        }
                    }
                    remaining = matchResult.suffix();
                }
                if (mixed) {
                    mixedPrecisionInfo[currentLineNumber] = info;
                }
            }
        }
        if (currentLineNumber == -1) {
            cerr << "No #line directives found, did you use a debug version of matc?" << endl;
            return;
        }
    }

    // Finally, dump out the annotated GLSL.

    for (int lineNumber : lineNumbers) {
        if (!glsl.count(lineNumber)) {
            continue;
        }
        istringstream ss(glsl.at(lineNumber));
        string glslCodeline;
        bool firstLine = true;
        while (getline(ss, glslCodeline, '\n')) {
            cout << glslCodeline;
            if (firstLine) {
                if (mixedPrecisionInfo.count(lineNumber)) {
                    string info = mixedPrecisionInfo.at(lineNumber);
                    cout << " // POTENTIAL MIXED PRECISION " + info;
                }
                firstLine = false;
            }
            cout << endl;
        }
    }
}

static void disassembleSpirv(const std::vector<uint32_t>& spirv, bool analyze) {
    // If desired feel free to locally replace this with the glslang disassembler (spv::Disassemble)
    // but please do not submit. We prefer to use the syntax that the standalone "spirv-dis" tool
    // uses, which lets us easily generate test cases for the spirv-cross project.
    auto context = spvContextCreate(SPV_ENV_UNIVERSAL_1_1);
    spv_text text = nullptr;
    const uint32_t options = SPV_BINARY_TO_TEXT_OPTION_INDENT |
            SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES;
    spvBinaryToText(context, spirv.data(), spirv.size(), options, &text, nullptr);
    if (analyze) {
        analyzeSpirv(spirv, text->str);
    } else {
        std::cout << text->str << std::endl;
    }
    spvTextDestroy(text);
    spvContextDestroy(context);
}

static void dumpSpirvBinary(const std::vector<uint32_t>& spirv, std::string filename) {
    std::ofstream out(filename, std::ofstream::binary);
    out.write((const char*) spirv.data(), spirv.size() * 4);
    std::cout << "Binary SPIR-V dumped to " << filename << std::endl;
}

static bool parseChunks(Config config, void* data, size_t size) {
    ChunkContainer container(data, size);
    if (!container.parse()) {
        return false;
    }
    if (config.printGLSL || config.printSPIRV || config.printMetal) {
        filaflat::ShaderBuilder builder;
        std::vector<ShaderInfo> info;

        if (config.printGLSL) {
            MaterialParser parser(Backend::OPENGL, data, size);
            if (!parser.parse() ||
                    (!parser.isShadingMaterial() && !parser.isPostProcessMaterial())) {
                return false;
            }

            if (!getGlShaderInfo(container, &info)) {
                std::cerr << "Failed to parse GLSL chunk." << std::endl;
                return false;
            }

            if (config.shaderIndex >= info.size()) {
                std::cerr << "Shader index out of range." << std::endl;
                return false;
            }

            const auto& item = info[config.shaderIndex];
            parser.getShader(item.shaderModel, item.variant, item.pipelineStage, builder);

            // Casted to char* to print as a string rather than hex value.
            std::cout << (const char*) builder.data();

            return true;
        }

        if (config.printSPIRV) {
            MaterialParser parser(Backend::VULKAN, data, size);
            if (!parser.parse() ||
                    (!parser.isShadingMaterial() && !parser.isPostProcessMaterial())) {
                return false;
            }

            if (!getVkShaderInfo(container, &info)) {
                std::cerr << "Failed to parse SPIRV chunk." << std::endl;
                return false;
            }

            if (config.shaderIndex >= info.size()) {
                std::cerr << "Shader index out of range." << std::endl;
                return false;
            }

            const auto& item = info[config.shaderIndex];
            parser.getShader(item.shaderModel, item.variant, item.pipelineStage, builder);

            // Build std::vector<uint32_t> since that's what the Khronos libraries consume.
            uint32_t const* words = reinterpret_cast<uint32_t const*>(builder.data());
            assert(0 == (builder.size() % 4));
            const std::vector<uint32_t> spirv(words, words + builder.size() / 4);

            if (config.transpile) {
                transpileSpirv(spirv);
            } else if (config.binary) {
                dumpSpirvBinary(spirv, "out.spv");
            } else {
                disassembleSpirv(spirv, config.analyze);
            }

            return true;
        }

        if (config.printMetal) {
            MaterialParser parser(Backend::METAL, data, size);
            if (!parser.parse() ||
                    (!parser.isShadingMaterial() && !parser.isPostProcessMaterial())) {
                return false;
            }

            if (!getMetalShaderInfo(container, &info)) {
                std::cerr << "Failed to parse Metal chunk." << std::endl;
                return false;
            }

            if (config.shaderIndex >= info.size()) {
                std::cerr << "Shader index out of range." << std::endl;
                return false;
            }

            const auto& item = info[config.shaderIndex];
            parser.getShader(item.shaderModel, item.variant, item.pipelineStage, builder);
            std::cout << (const char*) builder.data();

            return true;
        }
    }

    if (!printMaterialInfo(container)) {
        std::cerr << "The source material is invalid." << std::endl;
        return false;
    }

    return true;
}

// Parse the contents of .inc files, which look like: "0xba, 0xdf, 0xf0" etc. Happily, istream
// skips over whitespace and commas, and stoul takes care of leading "0x" when parsing hex.
static bool parseTextBlob(Config config, std::istream& in) {
    std::vector<char> buffer;
    std::string hexcode;
    while (in >> hexcode) {
        buffer.push_back(static_cast<char>(std::stoul(hexcode, nullptr, 16)));
    }
    return parseChunks(config, buffer.data(), buffer.size());
}

static bool parseBinary(Config config, std::istream& in, long fileSize) {
    std::vector<char> buffer(static_cast<unsigned long>(fileSize));
    if (in.read(buffer.data(), fileSize)) {
        return parseChunks(config, buffer.data(), buffer.size());
    }
    std::cerr << "Could not read the source material." << std::endl;
    return false;
}

int main(int argc, char* argv[]) {
    Config config;
    int optionIndex = handleArguments(argc, argv, &config);

    int numArgs = argc - optionIndex;
    if (numArgs < 1) {
        printUsage(argv[0]);
        return 1;
    }

    Path src(argv[optionIndex]);
    if (!src.exists()) {
        std::cerr << "The source material " << src << " does not exist." << std::endl;
        return 1;
    }

    long fileSize = static_cast<long>(getFileSize(src.c_str()));
    if (fileSize <= 0) {
        std::cerr << "The source material " << src << " is invalid." << std::endl;
        return 1;
    }

    std::ifstream in(src.c_str(), std::ifstream::in);
    if (in.is_open()) {
        if (src.getExtension() == "inc") {
            return parseTextBlob(config, in) ? 0 : 1;
        } else {
            return parseBinary(config, in, fileSize) ? 0 : 1;
        }
    } else {
        std::cerr << "Could not open the source material " << src << std::endl;
        return 1;
    };

    return 0;
}
