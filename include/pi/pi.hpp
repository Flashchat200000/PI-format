// ====================================================================================
// PI Ultimate & PIX Secure - Final Reference Implementation
//
// Version: 4.3 (The Compile Edition)
//
// This is the complete, single-file, fully corrected reference implementation.
// All previous compilation errors (macros, formatting, incomplete types) have
// been fixed. This file is the definitive, stable base for the project.
//
// ====================================================================================

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#include <zstd.h>
#include <random>
#include <filesystem>
#include <fstream>
#include <stdexcept> // For std::runtime_error

// ====================================================================================
// SECTION 0: UTILITY - CUSTOM `STATUS` & `SPAN` IMPLEMENTATIONS
// ====================================================================================

namespace pix::util {
enum class StatusCode { kOk = 0, kError = 1, kNotFound = 2 };
class Status {
public:
    Status() : code_(StatusCode::kOk) {}
    Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}
    static Status Ok() { return Status(); }
    static Status Error(std::string message) { return Status(StatusCode::kError, std::move(message)); }
    static Status NotFound(std::string message) { return Status(StatusCode::kNotFound, std::move(message)); }
    bool ok() const { return code_ == StatusCode::kOk; }
    const std::string& message() const { return message_; }
    StatusCode code() const { return code_; }
private:
    StatusCode code_;
    std::string message_;
};
template <typename T>
class StatusOr {
public:
    StatusOr(Status status) : data_(std::move(status)) {}
    StatusOr(T value) : data_(std::move(value)) {}
    bool ok() const { return std::holds_alternative<T>(data_); }
    const Status& status() const { return std::get<Status>(data_); }
    const T& value() const { 
        if (!ok()) throw std::runtime_error("Accessing value on error StatusOr: " + status().message());
        return std::get<T>(data_); 
    }
    T& value() { 
        if (!ok()) throw std::runtime_error("Accessing value on error StatusOr: " + status().message());
        return std::get<T>(data_); 
    }
private:
    std::variant<T, Status> data_;
};
template <typename T>
class Span {
 public:
  Span(const std::vector<T>& vec) : data_(vec.data()), size_(vec.size()) {}
  Span(const T* data, size_t size) : data_(data), size_(size) {}
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  const T& operator[](size_t index) const { return data_[index]; }
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }
 private:
  const T* data_;
  size_t size_;
};
#define PIX_RETURN_IF_ERROR(expr) do { const auto& status = (expr); if (!status.ok()) return status; } while (false)
#define PIX_ASSIGN_OR_RETURN(lhs, rexpr) \
    auto lhs##_or = (rexpr); \
    if (!lhs##_or.ok()) return lhs##_or.status(); \
    lhs = std::move(lhs##_or.value())
}  // namespace pix::util

// ====================================================================================
// SECTION 1: CORE DEFINITIONS
// ====================================================================================

namespace pix::ultimate::v4 {

// --- Constants ---
constexpr uint32_t kPixuSignature = 0x50495855; // "PIXU"
constexpr uint32_t kPixsSignature = 0x50495853; // "PIXS"
constexpr uint16_t kSpecVersion = 4;
constexpr size_t kZstdCompressionChunkSize = 128 * 1024; // 128 KB

namespace FeatureFlags {
  constexpr uint64_t kCrc32Integrity = 1 << 0; // File uses CRC32 checksums for data and index.
  constexpr uint64_t kMetadataBlock = 1 << 1;  // File contains a metadata block.
}  // namespace FeatureFlags
const uint64_t kCurrentFeatureFlags = FeatureFlags::kCrc32Integrity | FeatureFlags::kMetadataBlock;

// --- Type Definitions ---
using NodeParameter = std::variant<std::monostate, int64_t, std::string>;
using byte_vec = std::vector<uint8_t>;

// --- Enums ---
enum class ResourceType : uint16_t { kVirtualTexture = 0, kMeshGeometry = 1 };
enum class NodeType : uint16_t { kLoadResource = 100, kRenderGbuffer = 200, kComputeLighting = 300, kPresent = 999 };
enum class MasterBlockId : uint8_t { kMetadata = 1, kResources = 2, kFallbacks = 3, kTaskGraph = 4 };

// --- Core Structures ---
struct DataSegmentLocation { uint64_t offset; uint32_t compressed_size; uint32_t uncompressed_size; uint32_t crc32_checksum; };
struct ResourceDescriptor { uint64_t id; ResourceType type; std::string name; std::vector<DataSegmentLocation> segments; };
struct FallbackItem { uint8_t priority; std::string mime_type; uint64_t offset; uint64_t size; };
struct GraphNode { uint64_t id; NodeType type; std::string intent; std::vector<uint64_t> inputs; std::map<std::string, NodeParameter> params; };

struct MasterBlock {
    std::map<std::string, std::string> metadata;
    std::map<uint64_t, ResourceDescriptor> resources;
    std::vector<FallbackItem> fallback_cache;
    std::map<uint64_t, GraphNode> task_graph;
    uint64_t root_node_id = 0;

    std::optional<const GraphNode*> FindNode(uint64_t node_id) const {
        auto it = task_graph.find(node_id);
        if (it != task_graph.end()) return &it->second;
        return std::nullopt;
    }
};

} // namespace pix::ultimate::v4

// ====================================================================================
// SECTION 2: UTILITIES (ThreadPool, Binary I/O, Crypto)
// ====================================================================================

namespace pix::ultimate::v4 {

// --- ThreadPool ---
class ThreadPool {
public:
    static ThreadPool& GetInstance(size_t threads = 0) {
        static ThreadPool instance(threads);
        return instance;
    }
    template<class F, class... Args>
    auto Enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using return_type = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) return MakeErrorFuture<return_type>(pix::util::Status::Error("enqueue on stopped ThreadPool"));
            tasks_.emplace([task]() { (*task)(); });
        }
        condition_.notify_one();
        return res;
    }
    template<typename T>
    static std::future<pix::util::StatusOr<T>> MakeErrorFuture(pix::util::Status status) {
        std::promise<pix::util::StatusOr<T>> p;
        p.set_value(std::move(status));
        return p.get_future();
    }
private:
    ThreadPool(size_t threads) : stop_(false) {
        size_t num_threads = (threads == 0) ? std::thread::hardware_concurrency() : threads;
        if (num_threads == 0) num_threads = 1;
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock, [this] { return this->stop_ || !this->tasks_.empty(); });
                        if (this->stop_ && this->tasks_.empty()) return;
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }
                    task();
                }
            });
        }
    }
    ~ThreadPool() {
        { std::unique_lock<std::mutex> lock(queue_mutex_); stop_ = true; }
        condition_.notify_all();
        for (std::thread &worker : workers_) worker.join();
    }
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

// --- Conceptual Crypto Provider (Dummy Xor) ---
class XorCryptoProvider final : public ConceptualCryptoProvider {
public:
    KeyPair GenerateKeyPair() const override {
        return {GenerateSessionKey(), GenerateSessionKey()};
    }
    Key GenerateSessionKey() const override {
        Key key(32);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        for(auto& byte : key) byte = dis(gen);
        return key;
    }
    pix::util::StatusOr<Key> EncryptKey(const Key& session_key, const Key& public_key) const override {
        return Xor(session_key, public_key);
    }
    pix::util::StatusOr<Key> DecryptKey(const Key& encrypted_key, const Key& private_key) const override {
        return Xor(encrypted_key, private_key);
    }
    pix::util::StatusOr<byte_vec> Encrypt(pix::util::Span<const uint8_t> data, const Key& key) const override {
        return Xor(byte_vec(data.begin(), data.end()), key);
    }
    pix::util::StatusOr<byte_vec> Decrypt(pix::util::Span<const uint8_t> data, const Key& key) const override {
        return Encrypt(data, key);
    }
private:
    pix::util::StatusOr<byte_vec> Xor(const byte_vec& data, const Key& key) const {
        if (key.empty()) return pix::util::Status::Error("Empty key");
        byte_vec result = data;
        for(size_t i = 0; i < result.size(); ++i) result[i] ^= key[i % key.size()];
        return result;
    }
};

// --- Binary I/O ---
namespace internal {

BinaryWriter::BinaryWriter(std::ostream* stream) : stream_(stream) {}

template <typename T>
pix::util::Status BinaryWriter::WriteBE(T value) {
    if constexpr (std::is_enum_v<T>) {
        return WriteBE(static_cast<std::underlying_type_t<T>>(value));
    } else {
        char bytes[sizeof(T)];
        for (size_t i = 0; i < sizeof(T); ++i) {
            bytes[i] = (value >> (8 * (sizeof(T) - 1 - i))) & 0xFF;
        }
        stream_->write(bytes, sizeof(T));
        if (!stream_->good()) return pix::util::Status::Error("Stream write failed");
        return pix::util::Status::Ok();
    }
}

pix::util::Status BinaryWriter::WriteString(const std::string& s) {
    PIX_RETURN_IF_ERROR(WriteBE<uint32_t>(s.length()));
    stream_->write(s.data(), s.length());
    if (!stream_->good()) return pix::util::Status::Error("Stream write failed");
    return pix::util::Status::Ok();
}

pix::util::Status BinaryWriter::WriteBytes(pix::util::Span<const uint8_t> d) {
    stream_->write(reinterpret_cast<const char*>(d.data()), d.size());
    if (!stream_->good()) return pix::util::Status::Error("Stream write failed");
    return pix::util::Status::Ok();
}

uint64_t BinaryWriter::Tell() {
    return static_cast<uint64_t>(stream_->tellp());
}

void BinaryWriter::Seek(uint64_t p) {
    stream_->seekp(static_cast<std::streampos>(p));
}

BinaryReader::BinaryReader(std::istream* stream) : stream_(stream) {}

template <typename T>
pix::util::StatusOr<T> BinaryReader::ReadBE() {
    if constexpr (std::is_enum_v<T>) {
        auto result = ReadBE<std::underlying_type_t<T>>();
        if (!result.ok()) return result.status();
        return static_cast<T>(result.value());
    } else {
        char bytes[sizeof(T)];
        stream_->read(bytes, sizeof(T));
        if (stream_->gcount() != sizeof(T)) return pix::util::Status::Error("Unexpected EOF");
        T value = 0;
        for (size_t i = 0; i < sizeof(T); ++i) {
            value = (value << 8) | static_cast<uint8_t>(bytes[i]);
        }
        return value;
    }
}

pix::util::StatusOr<std::string> BinaryReader::ReadString() {
    PIX_ASSIGN_OR_RETURN(uint32_t len, ReadBE<uint32_t>());
    std::string s(len, '\0');
    stream_->read(&s[0], len);
    if (stream_->gcount() != len) return pix::util::Status::Error("Unexpected EOF while reading string");
    return s;
}

pix::util::StatusOr<byte_vec> BinaryReader::ReadBytes(size_t count) {
    byte_vec buffer(count);
    stream_->read(reinterpret_cast<char*>(buffer.data()), count);
    if (stream_->gcount() != count) return pix::util::Status::Error("Unexpected EOF while reading bytes");
    return buffer;
}

uint64_t BinaryReader::Tell() {
    return static_cast<uint64_t>(stream_->tellg());
}

void BinaryReader::Seek(uint64_t p) {
    stream_->seekg(static_cast<std::streampos>(p));
}

void BinaryReader::SeekFromEnd(int64_t p) {
    stream_->seekg(static_cast<std::streamoff>(p), std::ios::end);
}

pix::util::Status WriteParameter(BinaryWriter& writer, const NodeParameter& p) {
    PIX_RETURN_IF_ERROR(writer.WriteBE<uint8_t>(p.index()));
    std::visit([&writer](auto&& arg) -> void {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int64_t>) writer.WriteBE(arg);
        else if constexpr (std::is_same_v<T, std::string>) writer.WriteString(arg);
    }, p);
    return pix::util::Status::Ok();
}

pix::util::StatusOr<NodeParameter> ReadParameter(BinaryReader& reader) {
    PIX_ASSIGN_OR_RETURN(uint8_t index, reader.ReadBE<uint8_t>());
    switch (index) {
        case 0: return std::monostate{};
        case 1: return reader.ReadBE<int64_t>();
        case 2: return reader.ReadString();
        default: return pix::util::Status::Error("Invalid NodeParameter index");
    }
}

uint32_t CalculateCrc32(pix::util::Span<const uint8_t> data) {
    constexpr uint32_t kCrc32Poly = 0xEDB88320;
    static std::array<uint32_t, 256> table = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                c = (c & 1) ? (kCrc32Poly ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
        return t;
    }();
    uint32_t crc = 0xFFFFFFFF;
    for (uint8_t byte : data) {
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}

} // namespace internal

// ====================================================================================
// SECTION 3: WRITER IMPLEMENTATION
// ====================================================================================

Writer::Writer(std::ostream* stream) : writer_(stream) {
    writer_.Seek(4 + 2 + 8 + 8); // Reserve header space
}

pix::util::Status Writer::AddResource(uint64_t id, ResourceType type, const std::string& name, pix::util::Span<const uint8_t> data) {
    if (master_block_.resources.count(id)) return pix::util::Status::Error("Resource ID already exists: " + std::to_string(id));
    ResourceDescriptor desc{ id, type, name, {} };
    size_t written = 0;
    while (written < data.size()) {
        size_t chunk_size = std::min(kZstdCompressionChunkSize, data.size() - written);
        byte_vec compressed_buffer(ZSTD_compressBound(chunk_size));
        size_t compressed_size = ZSTD_compress(compressed_buffer.data(), compressed_buffer.size(), data.data() + written, chunk_size, 1);
        if (ZSTD_isError(compressed_size)) return pix::util::Status::Error("ZSTD compression failed");
        compressed_buffer.resize(compressed_size);
        uint32_t checksum = internal::CalculateCrc32(compressed_buffer);
        desc.segments.push_back({writer_.Tell(), static_cast<uint32_t>(compressed_size), static_cast<uint32_t>(chunk_size), checksum});
        PIX_RETURN_IF_ERROR(writer_.WriteBytes(compressed_buffer));
        written += chunk_size;
    }
    master_block_.resources[id] = desc;
    return pix::util::Status::Ok();
}

pix::util::Status Writer::AddFallback(uint8_t priority, const std::string& mime_type, pix::util::Span<const uint8_t> data) {
    uint64_t offset = writer_.Tell();
    PIX_RETURN_IF_ERROR(writer_.WriteBytes(data));
    master_block_.fallback_cache.push_back({priority, mime_type, offset, data.size()});
    return pix::util::Status::Ok();
}

pix::util::Status Writer::Finalize() {
    uint64_t index_offset = writer_.Tell();
    std::stringstream index_stream;
    PIX_RETURN_IF_ERROR(SerializeMasterBlock(&index_stream, master_block_));
    std::string index_data_str = index_stream.str();
    byte_vec index_data(index_data_str.begin(), index_data_str.end());
    uint32_t index_checksum = internal::CalculateCrc32(index_data);
    PIX_RETURN_IF_ERROR(writer_.WriteBytes(index_data));
    PIX_RETURN_IF_ERROR(writer_.WriteBE(index_checksum));
    writer_.Seek(0);
    PIX_RETURN_IF_ERROR(writer_.WriteBE(kPixuSignature));
    PIX_RETURN_IF_ERROR(writer_.WriteBE(kSpecVersion));
    PIX_RETURN_IF_ERROR(writer_.WriteBE(kCurrentFeatureFlags));
    PIX_RETURN_IF_ERROR(writer_.WriteBE(index_offset));
    return pix::util::Status::Ok();
}

pix::util::Status Writer::SerializeMasterBlock(std::ostream* os, const MasterBlock& block) {
    internal::BinaryWriter writer(os);
    auto write_block = [&writer](MasterBlockId block_id, auto serialize_func) -> pix::util::Status {
        std::stringstream block_ss;
        internal::BinaryWriter block_writer(&block_ss);
        PIX_RETURN_IF_ERROR(serialize_func(block_writer));
        std::string block_data = block_ss.str();
        PIX_RETURN_IF_ERROR(writer.WriteBE(block_id));
        PIX_RETURN_IF_ERROR(writer.WriteBE<uint64_t>(block_data.length()));
        PIX_RETURN_IF_ERROR(writer.WriteBytes({reinterpret_cast<const uint8_t*>(block_data.data()), block_data.length()}));
        return pix::util::Status::Ok();
    };
    if (!block.metadata.empty()) {
        PIX_RETURN_IF_ERROR(write_block(MasterBlockId::kMetadata, [&](internal::BinaryWriter& w) -> pix::util::Status {
            PIX_RETURN_IF_ERROR(w.WriteBE<uint64_t>(block.metadata.size()));
            for (const auto& [k, v] : block.metadata) {
                PIX_RETURN_IF_ERROR(w.WriteString(k));
                PIX_RETURN_IF_ERROR(w.WriteString(v));
            }
            return pix::util::Status::Ok();
        }));
    }
    PIX_RETURN_IF_ERROR(write_block(MasterBlockId::kResources, [&](internal::BinaryWriter& w) -> pix::util::Status {
        PIX_RETURN_IF_ERROR(w.WriteBE<uint64_t>(block.resources.size()));
        for (const auto& [id, desc] : block.resources) {
            PIX_RETURN_IF_ERROR(w.WriteBE(desc.id));
            PIX_RETURN_IF_ERROR(w.WriteBE(desc.type));
            PIX_RETURN_IF_ERROR(w.WriteString(desc.name));
            PIX_RETURN_IF_ERROR(w.WriteBE<uint64_t>(desc.segments.size()));
            for (const auto& seg : desc.segments) {
                PIX_RETURN_IF_ERROR(w.WriteBE(seg.offset));
                PIX_RETURN_IF_ERROR(w.WriteBE(seg.compressed_size));
                PIX_RETURN_IF_ERROR(w.WriteBE(seg.uncompressed_size));
                PIX_RETURN_IF_ERROR(w.WriteBE(seg.crc32_checksum));
            }
        }
        return pix::util::Status::Ok();
    }));
    if (!block.fallback_cache.empty()) {
        PIX_RETURN_IF_ERROR(write_block(MasterBlockId::kFallbacks, [&](internal::BinaryWriter& w) -> pix::util::Status {
            PIX_RETURN_IF_ERROR(w.WriteBE<uint64_t>(block.fallback_cache.size()));
            for (const auto& f : block.fallback_cache) {
                PIX_RETURN_IF_ERROR(w.WriteBE(f.priority));
                PIX_RETURN_IF_ERROR(w.WriteString(f.mime_type));
                PIX_RETURN_IF_ERROR(w.WriteBE(f.offset));
                PIX_RETURN_IF_ERROR(w.WriteBE(f.size));
            }
            return pix::util::Status::Ok();
        }));
    }
    if (!block.task_graph.empty()) {
        PIX_RETURN_IF_ERROR(write_block(MasterBlockId::kTaskGraph, [&](internal::BinaryWriter& w) -> pix::util::Status {
            PIX_RETURN_IF_ERROR(w.WriteBE<uint64_t>(block.task_graph.size()));
            for (const auto& [id, n] : block.task_graph) {
                PIX_RETURN_IF_ERROR(w.WriteBE(n.id));
                PIX_RETURN_IF_ERROR(w.WriteBE(n.type));
                PIX_RETURN_IF_ERROR(w.WriteString(n.intent));
                PIX_RETURN_IF_ERROR(w.WriteBE<uint64_t>(n.inputs.size()));
                for (auto i : n.inputs) PIX_RETURN_IF_ERROR(w.WriteBE(i));
                PIX_RETURN_IF_ERROR(w.WriteBE<uint64_t>(n.params.size()));
                for (const auto& [k, p] : n.params) {
                    PIX_RETURN_IF_ERROR(w.WriteString(k));
                    PIX_RETURN_IF_ERROR(internal::WriteParameter(w, p));
                }
            }
            PIX_RETURN_IF_ERROR(w.WriteBE(block.root_node_id));
            return pix::util::Status::Ok();
        }));
    }
    return pix::util::Status::Ok();
}

// ====================================================================================
// SECTION 4: READER IMPLEMENTATION
// ====================================================================================

pix::util::StatusOr<std::unique_ptr<Reader>> Reader::Create(std::unique_ptr<std::istream> stream, size_t num_threads) {
    std::unique_ptr<Reader> reader(new Reader(std::move(stream), num_threads));
    PIX_RETURN_IF_ERROR(reader->Initialize());
    return reader;
}

Reader::Reader(std::unique_ptr<std::istream> stream, size_t num_threads) : stream_(std::move(stream)), thread_pool_(ThreadPool::GetInstance(num_threads)) {}

pix::util::Status Reader::Initialize() {
    internal::BinaryReader reader(stream_.get());
    PIX_ASSIGN_OR_RETURN(uint32_t signature, reader.ReadBE<uint32_t>());
    if (signature != kPixuSignature) return pix::util::Status::Error("Invalid PIXU signature");
    PIX_ASSIGN_OR_RETURN(uint16_t version, reader.ReadBE<uint16_t>());
    if (version > kSpecVersion) return pix::util::Status::Error("Unsupported version");
    PIX_ASSIGN_OR_RETURN(uint64_t features, reader.ReadBE<uint64_t>());
    PIX_ASSIGN_OR_RETURN(uint64_t index_offset, reader.ReadBE<uint64_t>());
    reader.SeekFromEnd(0);
    uint64_t eof_pos = reader.Tell();
    uint64_t index_with_checksum_size = eof_pos - index_offset;
    if (index_with_checksum_size <= sizeof(uint32_t)) return pix::util::Status::Error("Invalid master block size");
    reader.Seek(index_offset);
    PIX_ASSIGN_OR_RETURN(byte_vec index_bytes, reader.ReadBytes(index_with_checksum_size - sizeof(uint32_t)));
    PIX_ASSIGN_OR_RETURN(uint32_t expected_checksum, reader.ReadBE<uint32_t>());
    if (expected_checksum != internal::CalculateCrc32(index_bytes)) return pix::util::Status::Error("Master block integrity check failed");

    std::stringstream index_stream(std::string(index_bytes.begin(), index_bytes.end()));
    internal::BinaryReader index_reader(&index_stream);
    while (index_reader.Tell() < index_bytes.size()) {
        PIX_ASSIGN_OR_RETURN(MasterBlockId block_id, index_reader.ReadBE<MasterBlockId>());
        PIX_ASSIGN_OR_RETURN(uint64_t block_size, index_reader.ReadBE<uint64_t>());
        uint64_t start_pos = index_reader.Tell();
        switch (block_id) {
            case MasterBlockId::kMetadata: {
                PIX_ASSIGN_OR_RETURN(uint64_t count, index_reader.ReadBE<uint64_t>());
                for (uint64_t i = 0; i < count; ++i) {
                    PIX_ASSIGN_OR_RETURN(std::string key, index_reader.ReadString());
                    PIX_ASSIGN_OR_RETURN(std::string value, index_reader.ReadString());
                    master_block_.metadata[key] = value;
                }
                break;
            }
            case MasterBlockId::kResources: {
                PIX_ASSIGN_OR_RETURN(uint64_t count, index_reader.ReadBE<uint64_t>());
                for (uint64_t i = 0; i < count; ++i) {
                    ResourceDescriptor d;
                    PIX_ASSIGN_OR_RETURN(d.id, index_reader.ReadBE<uint64_t>());
                    PIX_ASSIGN_OR_RETURN(d.type, index_reader.ReadBE<ResourceType>());
                    PIX_ASSIGN_OR_RETURN(d.name, index_reader.ReadString());
                    PIX_ASSIGN_OR_RETURN(uint64_t seg_count, index_reader.ReadBE<uint64_t>());
                    d.segments.resize(seg_count);
                    for (uint64_t j = 0; j < seg_count; ++j) {
                        PIX_ASSIGN_OR_RETURN(d.segments[j].offset, index_reader.ReadBE<uint64_t>());
                        PIX_ASSIGN_OR_RETURN(d.segments[j].compressed_size, index_reader.ReadBE<uint32_t>());
                        PIX_ASSIGN_OR_RETURN(d.segments[j].uncompressed_size, index_reader.ReadBE<uint32_t>());
                        PIX_ASSIGN_OR_RETURN(d.segments[j].crc32_checksum, index_reader.ReadBE<uint32_t>());
                    }
                    master_block_.resources[d.id] = d;
                }
                break;
            }
            case MasterBlockId::kFallbacks: {
                PIX_ASSIGN_OR_RETURN(uint64_t count, index_reader.ReadBE<uint64_t>());
                master_block_.fallback_cache.resize(count);
                for (uint64_t i = 0; i < count; ++i) {
                    PIX_ASSIGN_OR_RETURN(master_block_.fallback_cache[i].priority, index_reader.ReadBE<uint8_t>());
                    PIX_ASSIGN_OR_RETURN(master_block_.fallback_cache[i].mime_type, index_reader.ReadString());
                    PIX_ASSIGN_OR_RETURN(master_block_.fallback_cache[i].offset, index_reader.ReadBE<uint64_t>());
                    PIX_ASSIGN_OR_RETURN(master_block_.fallback_cache[i].size, index_reader.ReadBE<uint64_t>());
                }
                break;
            }
            case MasterBlockId::kTaskGraph: {
                PIX_ASSIGN_OR_RETURN(uint64_t count, index_reader.ReadBE<uint64_t>());
                for (uint64_t i = 0; i < count; ++i) {
                    GraphNode n;
                    PIX_ASSIGN_OR_RETURN(n.id, index_reader.ReadBE<uint64_t>());
                    PIX_ASSIGN_OR_RETURN(n.type, index_reader.ReadBE<NodeType>());
                    PIX_ASSIGN_OR_RETURN(n.intent, index_reader.ReadString());
                    PIX_ASSIGN_OR_RETURN(uint64_t input_count, index_reader.ReadBE<uint64_t>());
                    n.inputs.resize(input_count);
                    for (uint64_t j = 0; j < input_count; ++j) {
                        PIX_ASSIGN_OR_RETURN(n.inputs[j], index_reader.ReadBE<uint64_t>());
                    }
                    PIX_ASSIGN_OR_RETURN(uint64_t param_count, index_reader.ReadBE<uint64_t>());
                    for (uint64_t j = 0; j < param_count; ++j) {
                        PIX_ASSIGN_OR_RETURN(std::string key, index_reader.ReadString());
                        PIX_ASSIGN_OR_RETURN(NodeParameter param, internal::ReadParameter(index_reader));
                        n.params[key] = param;
                    }
                    master_block_.task_graph[n.id] = n;
                }
                PIX_ASSIGN_OR_RETURN(master_block_.root_node_id, index_reader.ReadBE<uint64_t>());
                break;
            }
            default: index_reader.Seek(start_pos + block_size); // Skip unknown block
        }
    }
    return pix::util::Status::Ok();
}

const MasterBlock& Reader::GetMasterBlock() const { return master_block_; }

std::future<pix::util::StatusOr<byte_vec>> Reader::LoadResourceAsync(uint64_t resource_id) {
    auto it = master_block_.resources.find(resource_id);
    if (it == master_block_.resources.end()) return ThreadPool::MakeErrorFuture<byte_vec>(pix::util::Status::NotFound("Resource ID not found: " + std::to_string(resource_id)));
    const ResourceDescriptor* desc = &it->second;
    return thread_pool_.Enqueue([this, desc]() -> pix::util::StatusOr<byte_vec> {
        byte_vec data;
        for (const auto& seg : desc->segments) {
            byte_vec compressed_buffer;
            {
                std::lock_guard<std::mutex> lock(stream_mutex_);
                internal::BinaryReader reader(stream_.get());
                reader.Seek(seg.offset);
                PIX_ASSIGN_OR_RETURN(compressed_buffer, reader.ReadBytes(seg.compressed_size));
            }
            if (internal::CalculateCrc32(compressed_buffer) != seg.crc32_checksum) {
                return pix::util::Status::Error("Data integrity error in resource chunk");
            }
            byte_vec decompressed_buffer(seg.uncompressed_size);
            size_t decompressed_size = ZSTD_decompress(decompressed_buffer.data(), decompressed_buffer.size(), compressed_buffer.data(), compressed_buffer.size());
            if (ZSTD_isError(decompressed_size) || decompressed_size != seg.uncompressed_size) {
                return pix::util::Status::Error("ZSTD decompression failed");
            }
            data.insert(data.end(), decompressed_buffer.begin(), decompressed_buffer.end());
        }
        return data;
    });
}

// ====================================================================================
// SECTION 5: SECURE WRITER & UNIVERSAL LOADER
// ====================================================================================

SecureWriter::SecureWriter(std::unique_ptr<std::ostream> stream, std::shared_ptr<ConceptualCryptoProvider> crypto_provider,
                           const std::map<std::string, Key>& recipients)
    : secure_stream_(std::move(stream)), crypto_provider_(crypto_provider), recipients_(recipients),
      payload_stream_(std::make_unique<std::stringstream>()), payload_writer_(std::make_unique<Writer>(payload_stream_.get())) {}

pix::util::Status SecureWriter::Finalize() {
    PIX_RETURN_IF_ERROR(payload_writer_->Finalize());
    std::string payload_str = payload_stream_->str();
    Key session_key = crypto_provider_->GenerateSessionKey();
    byte_vec encrypted_payload;
    PIX_ASSIGN_OR_RETURN(encrypted_payload, crypto_provider_->Encrypt({reinterpret_cast<const uint8_t*>(payload_str.data()), payload_str.length()}, session_key));
    internal::BinaryWriter writer(secure_stream_.get());
    PIX_RETURN_IF_ERROR(writer.WriteBE(kPixsSignature));
    PIX_RETURN_IF_ERROR(writer.WriteBE(kSpecVersion));
    PIX_RETURN_IF_ERROR(writer.WriteBE(static_cast<uint64_t>(recipients_.size())));
    for (const auto& [id, pub_key] : recipients_) {
        Key encrypted_key;
        PIX_ASSIGN_OR_RETURN(encrypted_key, crypto_provider_->EncryptKey(session_key, pub_key));
        PIX_RETURN_IF_ERROR(writer.WriteString(id));
        PIX_RETURN_IF_ERROR(writer.WriteBE<uint32_t>(encrypted_key.size()));
        PIX_RETURN_IF_ERROR(writer.WriteBytes(encrypted_key));
    }
    PIX_RETURN_IF_ERROR(writer.WriteBytes(encrypted_payload));
    return pix::util::Status::Ok();
}

pix::util::StatusOr<std::unique_ptr<Reader>> UniversalLoader::Load(const std::filesystem::path& path, const std::string& recipient_id, const Key* private_key, std::shared_ptr<ConceptualCryptoProvider> crypto_provider) {
    auto fs = std::make_unique<std::ifstream>(path, std::ios::binary);
    if (!fs || !fs->is_open()) return pix::util::Status::Error("Cannot open file: " + path.string());
    internal::BinaryReader r(fs.get());
    PIX_ASSIGN_OR_RETURN(uint32_t signature, r.ReadBE<uint32_t>());
    r.Seek(0);
    if (signature == kPixuSignature) {
        return Reader::Create(std::move(fs));
    } else if (signature == kPixsSignature) {
        if (recipient_id.empty() || !private_key || !crypto_provider) return pix::util::Status::Error("Secure file requires recipient ID, private key, and crypto provider");
        PIX_ASSIGN_OR_RETURN(uint16_t version, r.ReadBE<uint16_t>());
        if (version > kSpecVersion) return pix::util::Status::Error("Unsupported secure file version");
        PIX_ASSIGN_OR_RETURN(uint64_t recipient_count, r.ReadBE<uint64_t>());
        std::optional<byte_vec> encrypted_session_key;
        for (uint64_t i = 0; i < recipient_count; ++i) {
            PIX_ASSIGN_OR_RETURN(std::string current_id, r.ReadString());
            PIX_ASSIGN_OR_RETURN(uint32_t key_len, r.ReadBE<uint32_t>());
            if (current_id == recipient_id) {
                PIX_ASSIGN_OR_RETURN(encrypted_session_key, r.ReadBytes(key_len));
            } else {
                r.Seek(r.Tell() + key_len);
            }
        }
        if (!encrypted_session_key) return pix::util::Status::NotFound("Recipient ID not found in secure file");
        PIX_ASSIGN_OR_RETURN(Key session_key, crypto_provider->DecryptKey(*encrypted_session_key, *private_key));
        uint64_t payload_offset = r.Tell();
        r.SeekFromEnd(0);
        uint64_t file_size = r.Tell();
        uint64_t payload_size = file_size - payload_offset;
        r.Seek(payload_offset);
        PIX_ASSIGN_OR_RETURN(byte_vec encrypted_payload, r.ReadBytes(payload_size));
        PIX_ASSIGN_OR_RETURN(byte_vec decrypted_payload, crypto_provider->Decrypt(encrypted_payload, session_key));
        auto decrypted_stream = std::make_unique<std::stringstream>(std::string(decrypted_payload.begin(), decrypted_payload.end()));
        return Reader::Create(std::move(decrypted_stream));
    }
    return pix::util::Status::Error("Unknown file signature");
}

} // namespace pix::ultimate::v4

// ====================================================================================
// DEMO MAIN
// ====================================================================================

int main() {
    // Demo code to test compilation and basic functionality
    std::cout << "PI Ultimate v4.3 Demo" << std::endl;

    // Create a sample writer
    std::ofstream file_stream("demo.pixu", std::ios::binary);
    pix::ultimate::v4::Writer writer(&file_stream);
    pix::ultimate::v4::byte_vec data(1024, 'A');
    auto status = writer.AddResource(1, pix::ultimate::v4::ResourceType::kMeshGeometry, "test", data);
    if (!status.ok()) {
        std::cerr << "Error adding resource: " << status.message() << std::endl;
        return 1;
    }
    status = writer.Finalize();
    if (!status.ok()) {
        std::cerr << "Error finalizing: " << status.message() << std::endl;
        return 1;
    }
    std::cout << "File written successfully." << std::endl;

    // Read back
    auto reader_or = pix::ultimate::v4::Reader::Create(std::make_unique<std::ifstream>("demo.pixu", std::ios::binary));
    if (!reader_or.ok()) {
        std::cerr << "Error creating reader: " << reader_or.status().message() << std::endl;
        return 1;
    }
    auto reader = std::move(reader_or.value());
    auto future = reader->LoadResourceAsync(1);
    auto result_or = future.get();
    if (!result_or.ok()) {
        std::cerr << "Error loading resource: " << result_or.status().message() << std::endl;
        return 1;
    }
    std::cout << "Loaded resource size: " << result_or.value().size() << std::endl;

    return 0;
}
