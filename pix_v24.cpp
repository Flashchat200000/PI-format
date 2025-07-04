// ====================================================================================
// PIX Format v24 - "The Declarative Web Container"
// Complete, Standalone, Professional C++ Reference Implementation
//
// This file contains the full, compilable source code for the v24 specification.
// It integrates all previously discussed concepts into a single, cohesive unit.
// ====================================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <filesystem>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <variant>
#include <optional>

namespace pix::v24 {

// ====================================================================================
// SECTION 1: CORE DEFINITIONS & DATA STRUCTURES
// ====================================================================================

// --- Format Constants ---
constexpr uint32_t PIX_SIGNATURE = 0x50495846; // "PIXF"
constexpr uint32_t INDEX_POINTER_MAGIC = 0xDEADBEEF;
constexpr uint64_t SCENE_TREE_ID = 1;

// --- Core Enumerations ---
enum class ChunkType : uint32_t {
    FBCK = 0x4642434B, TREE = 0x54524545, NODE = 0x4E4F4445, ACRV = 0x41435256, SCPT = 0x53435054, INDX = 0x494E4458,
};
enum class LayerType : uint8_t { GROUP = 0, PROCEDURAL = 1, RASTER = 2 };
enum class NodeType : uint32_t {
    NOISE_GENERATOR = 1, BLUR_FILTER = 2, TRANSFORM = 3, GRADIENT_MAP = 4,
    RASTER_IMAGE = 100, MERGE = 200, OUTPUT = 201,
};
namespace NodeParamID {
    constexpr uint32_t IMAGE_FORMAT = 1, IMAGE_DATA = 2, OPACITY = 10;
}

// --- Custom Exceptions ---
class PIXException : public std::runtime_error { public: using std::runtime_error::runtime_error; };
class InvalidFileFormatError : public PIXException { public: using PIXException::PIXException; };
class ChunkCorruptionError : public PIXException { public: using PIXException::PIXException; };

// --- Data Structures ---
struct Vec2f { float x = 0.0f, y = 0.0f; };
struct Keyframe { float time; float value; Vec2f handle_in; Vec2f handle_out; };
using NodeParameter = std::variant<float, int32_t, std::string, std::vector<char>>;
struct TreeNode { uint64_t object_id; uint64_t parent_id; LayerType layer_type; };
struct Node { uint64_t node_id; NodeType type; std::vector<uint64_t> input_node_ids; std::map<uint32_t, NodeParameter> parameters; };
struct AnimationCurve { uint64_t curve_id; uint64_t target_object_id; uint32_t target_property_id; std::vector<Keyframe> keys; };
struct Script { uint64_t script_id; uint64_t attached_to_object_id; std::string code; };
struct FallbackCache { std::string format; std::vector<char> data; };
struct IndexEntry { uint64_t offset; uint32_t length; };
using IndexMap = std::map<uint64_t, IndexEntry>;

// ====================================================================================
// SECTION 2: UTILITY IMPLEMENTATIONS (BINARY I/O, CRC32)
// ====================================================================================
namespace util {
    uint32_t crc32_table[256];
    void build_crc32_table() {
        static bool is_built = false; if (is_built) return;
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i; for (size_t j = 0; j < 8; ++j) c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
            crc32_table[i] = c;
        } is_built = true;
    }

    uint32_t calculate_crc32(const char* data, size_t length) {
        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < length; ++i) crc = crc32_table[(crc ^ static_cast<uint8_t>(data[i])) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFF;
    }

    template<typename T> void write_be(std::ostream& os, T value) {
        char buffer[sizeof(T)];
        if constexpr (std::is_floating_point_v<T>) {
            // For floating-point types, reinterpret as integer for bitwise operations
            if constexpr (sizeof(T) == sizeof(uint32_t)) {
                uint32_t int_val = *reinterpret_cast<uint32_t*>(&value);
                for (size_t i = 0; i < sizeof(T); ++i) buffer[i] = static_cast<char>((int_val >> (8 * (sizeof(T) - 1 - i))) & 0xFF);
            } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
                uint64_t int_val = *reinterpret_cast<uint64_t*>(&value);
                for (size_t i = 0; i < sizeof(T); ++i) buffer[i] = static_cast<char>((int_val >> (8 * (sizeof(T) - 1 - i))) & 0xFF);
            }
        } else {
            // For integer types, direct bitwise operations
            for (size_t i = 0; i < sizeof(T); ++i) buffer[i] = static_cast<char>((static_cast<uint64_t>(value) >> (8 * (sizeof(T) - 1 - i))) & 0xFF);
        }
        os.write(buffer, sizeof(T));
    }

    template<typename T> T read_be(std::istream& is) {
        T value = 0; char buffer[sizeof(T)]; is.read(buffer, sizeof(T));
        if (static_cast<size_t>(is.gcount()) != sizeof(T)) throw InvalidFileFormatError("Unexpected EOF while reading a value.");
        if constexpr (std::is_floating_point_v<T>) {
            if constexpr (sizeof(T) == sizeof(uint32_t)) {
                uint32_t int_val = 0;
                for (size_t i = 0; i < sizeof(T); ++i) int_val |= static_cast<uint32_t>(static_cast<uint8_t>(buffer[i])) << (8 * (sizeof(T) - 1 - i));
                value = *reinterpret_cast<T*>(&int_val);
            } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
                uint64_t int_val = 0;
                for (size_t i = 0; i < sizeof(T); ++i) int_val |= static_cast<uint64_t>(static_cast<uint8_t>(buffer[i])) << (8 * (sizeof(T) - 1 - i));
                value = *reinterpret_cast<T*>(&int_val);
            }
        } else {
            for (size_t i = 0; i < sizeof(T); ++i) value |= static_cast<T>(static_cast<uint8_t>(buffer[i])) << (8 * (sizeof(T) - 1 - i));
        }
        return value;
    }

    void write_string(std::ostream& s, const std::string& str) {
        write_be<uint16_t>(s, static_cast<uint16_t>(str.size())); s.write(str.data(), str.size());
    }

    std::string read_string(std::istream& s) {
        uint16_t len = read_be<uint16_t>(s); std::string str(len, '\0'); s.read(&str[0], len);
        if (static_cast<size_t>(s.gcount()) != len) throw InvalidFileFormatError("Unexpected EOF while reading a string.");
        return str;
    }

    void write_blob(std::ostream& s, const std::vector<char>& blob) {
        write_be<uint32_t>(s, static_cast<uint32_t>(blob.size())); s.write(blob.data(), blob.size());
    }

    std::vector<char> read_blob(std::istream& s) {
        uint32_t len = read_be<uint32_t>(s); std::vector<char> blob(len); s.read(blob.data(), len);
        if (static_cast<size_t>(s.gcount()) != len) throw InvalidFileFormatError("Unexpected EOF while reading a blob.");
        return blob;
    }
} // namespace util

// ====================================================================================
// SECTION 3: CHUNK SERIALIZATION ARCHITECTURE
// ====================================================================================
class BaseChunk {
public:
    virtual ~BaseChunk() = default;
    virtual ChunkType type() const = 0;
    void write_to(std::ostream& stream) const {
        std::stringstream payload_stream(std::ios::out | std::ios::binary);
        serialize_payload(payload_stream);
        std::string payload_str = payload_stream.str();
        uint32_t type_val = static_cast<uint32_t>(this->type());
        std::vector<char> crc_buffer(4 + payload_str.size());
        for(size_t i=0; i<4; ++i) crc_buffer[i] = (type_val >> (8*(3-i))) & 0xFF;
        if (!payload_str.empty()) std::copy(payload_str.begin(), payload_str.end(), crc_buffer.begin() + 4);
        uint32_t crc = util::calculate_crc32(crc_buffer.data(), crc_buffer.size());
        util::write_be(stream, static_cast<uint32_t>(payload_str.size()));
        util::write_be(stream, type_val);
        stream.write(payload_str.data(), payload_str.size());
        util::write_be(stream, crc);
    }
protected:
    virtual void serialize_payload(std::ostream& os) const = 0;
};

class FBCKChunk : public BaseChunk {
    const FallbackCache& _cache;
public:
    explicit FBCKChunk(const FallbackCache& cache) : _cache(cache) {}
    ChunkType type() const override { return ChunkType::FBCK; }
protected:
    void serialize_payload(std::ostream& os) const override {
        util::write_string(os, _cache.format); util::write_blob(os, _cache.data);
    }
};

class TREEChunk : public BaseChunk {
    const std::vector<TreeNode>& _tree;
public:
    explicit TREEChunk(const std::vector<TreeNode>& tree) : _tree(tree) {}
    ChunkType type() const override { return ChunkType::TREE; }
protected:
    void serialize_payload(std::ostream& os) const override {
        util::write_be<uint32_t>(os, static_cast<uint32_t>(_tree.size()));
        for(const auto& n : _tree) { util::write_be(os, n.object_id); util::write_be(os, n.parent_id); util::write_be<uint8_t>(os, static_cast<uint8_t>(n.layer_type)); }
    }
};

class NODEChunk : public BaseChunk {
    const Node& _node;
public:
    explicit NODEChunk(const Node& n) : _node(n) {}
    ChunkType type() const override { return ChunkType::NODE; }
protected:
    void serialize_payload(std::ostream& os) const override {
        util::write_be(os, _node.node_id); util::write_be(os, static_cast<uint32_t>(_node.type));
        util::write_be<uint16_t>(os, static_cast<uint16_t>(_node.input_node_ids.size())); for(auto id : _node.input_node_ids) util::write_be(os, id);
        util::write_be<uint16_t>(os, static_cast<uint16_t>(_node.parameters.size()));
        for(const auto& [id, p] : _node.parameters) {
            util::write_be(os, id); util::write_be<uint8_t>(os, static_cast<uint8_t>(p.index()));
            std::visit([&](auto&& arg){
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, float>) util::write_be(os, arg);
                else if constexpr (std::is_same_v<T, int32_t>) util::write_be(os, arg);
                else if constexpr (std::is_same_v<T, std::string>) util::write_string(os, arg);
                else if constexpr (std::is_same_v<T, std::vector<char>>) util::write_blob(os, arg);
            }, p);
        }
    }
};

class ACRVChunk : public BaseChunk {
    const AnimationCurve& _curve;
public:
    explicit ACRVChunk(const AnimationCurve& c) : _curve(c) {}
    ChunkType type() const override { return ChunkType::ACRV; }
protected:
    void serialize_payload(std::ostream& os) const override {
        util::write_be(os, _curve.curve_id); util::write_be(os, _curve.target_object_id); util::write_be(os, _curve.target_property_id);
        util::write_be<uint32_t>(os, static_cast<uint32_t>(_curve.keys.size()));
        for(const auto& k : _curve.keys) { util::write_be(os, k.time); util::write_be(os, k.value); util::write_be(os, k.handle_in.x); util::write_be(os, k.handle_in.y); util::write_be(os, k.handle_out.x); util::write_be(os, k.handle_out.y); }
    }
};

class SCPTChunk : public BaseChunk {
    const Script& _script;
public:
    explicit SCPTChunk(const Script& s) : _script(s) {}
    ChunkType type() const override { return ChunkType::SCPT; }
protected:
    void serialize_payload(std::ostream& os) const override {
        util::write_be(os, _script.script_id); util::write_be(os, _script.attached_to_object_id);
        util::write_string(os, _script.code); // Changed to write_string for consistency
    }
};

class INDXChunk : public BaseChunk {
    const IndexMap& _map;
public:
    explicit INDXChunk(const IndexMap& map) : _map(map) {}
    ChunkType type() const override { return ChunkType::INDX; }
protected:
    void serialize_payload(std::ostream& os) const override {
        util::write_be<uint32_t>(os, static_cast<uint32_t>(_map.size()));
        for (const auto& [id, entry] : _map) { util::write_be<uint64_t>(os, id); util::write_be<uint64_t>(os, entry.offset); util::write_be<uint32_t>(os, entry.length); } 
    }
};

// ====================================================================================
// SECTION 4: SCENE ACCESS LAYER (LAZY LOADING)
// ====================================================================================
class Scene {
public:
    explicit Scene(const std::filesystem::path& filepath) : _filepath(filepath.string()) {
        _file.open(_filepath, std::ios::binary); if (!_file) throw PIXException("Cannot open file: " + _filepath);
        if (util::read_be<uint32_t>(_file) != PIX_SIGNATURE) throw InvalidFileFormatError("Invalid PIX signature.");
        _fallback_cache = _read_fallback_chunk();
        _load_master_index();
        std::cout << "[Scene] Smart client initialized. Fallback cache loaded and master index parsed with " << _index.size() << " dynamic objects." << std::endl;
    }

    const FallbackCache& get_fallback_cache() const { return _fallback_cache; }
    const IndexMap& get_index() const { return _index; }

    // Placeholder implementations for lazy loading of scene components
    std::optional<std::vector<TreeNode>> get_tree() const {
        return _get_or_load_object(SCENE_TREE_ID, _tree_cache, [](std::stringstream& s) {
            std::vector<TreeNode> tree;
            uint32_t count = util::read_be<uint32_t>(s);
            for(uint32_t i=0; i < count; ++i) {
                TreeNode n;
                n.object_id = util::read_be<uint64_t>(s);
                n.parent_id = util::read_be<uint64_t>(s);
                n.layer_type = static_cast<LayerType>(util::read_be<uint8_t>(s));
                tree.push_back(n);
            }
            return tree;
        });
    }

    std::optional<Node> get_node(uint64_t id) const {
        return _get_or_load_object(id, _node_cache, [](std::stringstream& s) {
            Node node;
            node.node_id = util::read_be<uint64_t>(s);
            node.type = static_cast<NodeType>(util::read_be<uint32_t>(s));
            uint16_t input_count = util::read_be<uint16_t>(s);
            for(uint16_t i=0; i < input_count; ++i) node.input_node_ids.push_back(util::read_be<uint64_t>(s));
            uint16_t param_count = util::read_be<uint16_t>(s);
            for(uint16_t i=0; i < param_count; ++i) {
                uint32_t param_id = util::read_be<uint32_t>(s);
                uint8_t type_idx = util::read_be<uint8_t>(s);
                if (type_idx == 0) node.parameters[param_id] = util::read_be<float>(s);
                else if (type_idx == 1) node.parameters[param_id] = util::read_be<int32_t>(s);
                else if (type_idx == 2) node.parameters[param_id] = util::read_string(s);
                else if (type_idx == 3) node.parameters[param_id] = util::read_blob(s);
            }
            return node;
        });
    }

    std::optional<AnimationCurve> get_animation_curve(uint64_t id) const {
        return _get_or_load_object(id, _curve_cache, [](std::stringstream& s) {
            AnimationCurve curve;
            curve.curve_id = util::read_be<uint64_t>(s);
            curve.target_object_id = util::read_be<uint64_t>(s);
            curve.target_property_id = util::read_be<uint32_t>(s);
            uint32_t key_count = util::read_be<uint32_t>(s);
            for(uint32_t i=0; i < key_count; ++i) {
                Keyframe k;
                k.time = util::read_be<float>(s);
                k.value = util::read_be<float>(s);
                k.handle_in.x = util::read_be<float>(s);
                k.handle_in.y = util::read_be<float>(s);
                k.handle_out.x = util::read_be<float>(s);
                k.handle_out.y = util::read_be<float>(s);
                curve.keys.push_back(k);
            }
            return curve;
        });
    }

    std::optional<Script> get_script(uint64_t id) const {
        return _get_or_load_object(id, _script_cache, [](std::stringstream& s) {
            Script scr;
            scr.script_id = util::read_be<uint64_t>(s);
            scr.attached_to_object_id = util::read_be<uint64_t>(s);
            scr.code = util::read_string(s); // Changed to read_string
            return scr;
        });
    }

private:
    std::string _filepath; mutable std::ifstream _file; FallbackCache _fallback_cache; IndexMap _index;
    mutable std::optional<std::vector<TreeNode>> _tree_cache; mutable std::map<uint64_t, Node> _node_cache;
    mutable std::map<uint64_t, AnimationCurve> _curve_cache; mutable std::map<uint64_t, Script> _script_cache;

    std::tuple<uint32_t, uint32_t, std::vector<char>, uint32_t> _read_chunk_at_current_pos() const {
        uint32_t payload_len = util::read_be<uint32_t>(_file); uint32_t type_val = util::read_be<uint32_t>(_file);
        std::vector<char> payload(payload_len); if (payload_len > 0) { _file.read(payload.data(), payload.size()); if (static_cast<size_t>(_file.gcount()) != payload_len) throw InvalidFileFormatError("Unexpected EOF reading chunk payload."); }
        uint32_t crc_from_file = util::read_be<uint32_t>(_file);
        std::vector<char> crc_buffer(4 + payload_len); for(size_t i=0; i<4; ++i) crc_buffer[i] = (type_val >> (8*(3-i))) & 0xFF;
        if (!payload.empty()) std::copy(payload.begin(), payload.end(), crc_buffer.begin() + 4);
        uint32_t calculated_crc = util::calculate_crc32(crc_buffer.data(), crc_buffer.size());
        if (crc_from_file != calculated_crc) { std::stringstream ss; ss << "CRC mismatch for chunk type 0x" << std::hex << type_val; throw ChunkCorruptionError(ss.str()); } 
        return { payload_len, type_val, payload, crc_from_file };
    }

    FallbackCache _read_fallback_chunk() const {
        _file.seekg(4); // Skip PIX_SIGNATURE
        auto [len, type, payload, crc] = _read_chunk_at_current_pos();
        if (static_cast<ChunkType>(type) != ChunkType::FBCK) throw InvalidFileFormatError("First chunk is not FBCK. Not a valid PIX v24 file.");
        std::stringstream s(std::string(payload.begin(), payload.end())); FallbackCache cache; cache.format = util::read_string(s);
        cache.data = util::read_blob(s); // Changed to read_blob
        return cache;
    }

    void _load_master_index() {
        _file.seekg(-static_cast<std::streamoff>(sizeof(uint64_t) + sizeof(uint32_t)), std::ios::end);
        uint64_t index_offset = util::read_be<uint64_t>(_file); uint32_t magic = util::read_be<uint32_t>(_file);
        if (magic != INDEX_POINTER_MAGIC) throw InvalidFileFormatError("Invalid index pointer magic.");
        _file.seekg(index_offset);
        auto [len, type, payload, crc] = _read_chunk_at_current_pos();
        if (static_cast<ChunkType>(type) != ChunkType::INDX) throw InvalidFileFormatError("Index pointer does not point to an INDX chunk.");
        std::stringstream s(std::string(payload.begin(), payload.end())); uint32_t count = util::read_be<uint32_t>(s);
        for(uint32_t i=0; i < count; ++i) _index[util::read_be<uint64_t>(s)] = {util::read_be<uint64_t>(s), util::read_be<uint32_t>(s)};
    }

    std::optional<std::vector<char>> _load_payload_by_id(uint64_t id) const {
        if (!_index.count(id)) return std::nullopt; const auto& entry = _index.at(id);
        std::cout << "[DISK] Loading object ID " << id << " from offset " << entry.offset << "..." << std::endl;
        _file.seekg(entry.offset); auto [len, type, payload, crc] = _read_chunk_at_current_pos(); return payload;
    }

    template<typename T, typename Cache, typename Parser>
    std::optional<T> _get_or_load_object(uint64_t id, Cache& cache, Parser parser) const {
        if (cache.count(id)) { std::cout << "[CACHE] Object ID " << id << " retrieved from cache." << std::endl; return cache.at(id); }
        auto payload = _load_payload_by_id(id); if (!payload) return std::nullopt;
        std::stringstream s(std::string(payload->begin(), payload->end()));
        T object = parser(s); cache[id] = object; return object;
    }
};

// ====================================================================================
// SECTION 5: SCENE BUILDER API
// ====================================================================================
class SceneBuilder {
public:
    explicit SceneBuilder(FallbackCache cache) : _fallback_cache(std::move(cache)) {}
    SceneBuilder& set_tree(std::vector<TreeNode> tree) { _tree = std::move(tree); return *this; }
    SceneBuilder& add_node(Node node) { _nodes[node.node_id] = std::move(node); return *this; }
    SceneBuilder& add_animation_curve(AnimationCurve curve) { _curves[curve.curve_id] = std::move(curve); return *this; }
    SceneBuilder& add_script(Script script) { _scripts[script.script_id] = std::move(script); return *this; }
    void write_to_file(const std::filesystem::path& filepath) const {
        std::cout << "\n--- Building and writing scene to 

void run_demonstration() {
    util::build_crc32_table();
    const std::filesystem::path scene_file = "declarative_web_container_v24.pix";
    const std::filesystem::path static_image_file = "preview.webp";
    try {
        create_dummy_webp(static_image_file);
        FallbackCache fallback = { "webp", read_file_to_blob(static_image_file) };
        SceneBuilder builder(fallback);
        builder.add_node(Node{101, NodeType::RASTER_IMAGE, {}, {{NodeParamID::IMAGE_DATA, fallback.data}}})
               .add_script(Script{501, 101, "PIX.getNode(101).setParameter(\'opacity\', Math.sin(PIX.time * 2.0) * 0.5 + 0.5);"});
        builder.write_to_file(scene_file);
        
        std::cout << "\n--- Demonstrating dual-mode access ---" << std::endl;
        std::cout << "\n1. \'Dumb\' Client Simulation:" << std::endl;
        std::vector<char> fallback_data = extract_fallback_image(scene_file);
        std::cout << "--> SUCCESS: Extracted " << fallback_data.size() << " bytes of fallback image data." << std::endl;
        
        std::cout << "\n2. \'Smart\' Client Simulation:" << std::endl;
        Scene scene(scene_file);
        auto script = scene.get_script(501);
        if (script) std::cout << "--> SUCCESS: Loaded interactive script: \"" << script->code << "\"" << std::endl;
    } catch (const std::exception& e) { std::cerr << "\nAN ERROR OCCURRED: " << e.what() << std::endl; }
}
} // namespace pix::v24

int main() {
    pix::v24::run_demonstration();
    std::cout << "\n--- End of Demonstration ---" << std::endl;
    return 0;
}

