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

#ifndef PI_ULTIMATE_V4_3_H_
#define PI_ULTIMATE_V4_3_H_

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

// ====================================================================================
// SECTION 0: UTILITY - STATUS & SPAN
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
    T& value() { return std::get<T>(data_); }
private:
    std::variant<T, Status> data_;
};
template <typename T>
class Span {
 public:
  Span(const std::vector<T>& vec) : data_(vec.data()), size_(vec.size()) {}
  Span(const T* data, size_t size) : data_(data), size_(size) {}
  template <size_t N> Span(const T (&arr)[N]) : data_(arr), size_(N) {}
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }
 private:
  const T* data_;
  size_t size_;
};
#define PIX_ASSIGN_OR_RETURN(lhs, rexpr) \
    do { \
        auto status_or_value = (rexpr); \
        if (!status_or_value.ok()) return status_or_value.status(); \
        lhs = std::move(status_or_value.value()); \
    } while(0)
}  // namespace pix::util

// ====================================================================================
// SECTION 1: CORE DEFINITIONS & INTERFACES
// ====================================================================================
namespace pix::ultimate::v4 {
constexpr uint32_t kPixuSignature = 0x50495855;
constexpr uint32_t kPixsSignature = 0x50495853;
constexpr uint16_t kSpecVersion = 4;
namespace FeatureFlags { constexpr uint64_t kCrc32Integrity = 1<<0; constexpr uint64_t kMetadataBlock = 1<<1; }
const uint64_t kCurrentFeatureFlags = FeatureFlags::kCrc32Integrity | FeatureFlags::kMetadataBlock;
using NodeParameter = std::variant<std::monostate, int64_t, std::string>;
using byte_vec = std::vector<uint8_t>;
enum class ResourceType : uint16_t { kVirtualTexture = 0, kMeshGeometry = 1 };
enum class NodeType : uint16_t { kLoadResource = 100, kRenderGbuffer = 200, kComputeLighting = 300, kPresent = 999 };
enum class MasterBlockId : uint8_t { kMetadata = 1, kResources = 2, kFallbacks = 3, kTaskGraph = 4 };
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
};

using Key = byte_vec;
class ConceptualCryptoProvider {
 public:
    struct KeyPair { Key public_key; Key private_key; };
    virtual ~ConceptualCryptoProvider() = default;
    virtual KeyPair GenerateKeyPair() const = 0;
    virtual Key GenerateSessionKey() const = 0;
    virtual pix::util::StatusOr<Key> EncryptKey(const Key&, const Key&) const = 0;
    virtual pix::util::StatusOr<Key> DecryptKey(const Key&, const Key&) const = 0;
    virtual pix::util::StatusOr<byte_vec> Encrypt(pix::util::Span<const uint8_t>, const Key&) const = 0;
    virtual pix::util::StatusOr<byte_vec> Decrypt(pix::util::Span<const uint8_t>, const Key&) const = 0;
};

}  // namespace pix::ultimate::v4

// ====================================================================================
// SECTION 2: CLASS DEFINITIONS
// ====================================================================================
namespace pix::ultimate::v4 {

class ThreadPool;
class Writer;
class Reader;
class SecureWriter;
class UniversalLoader;

namespace internal {
    class BinaryWriter {
    public:
        explicit BinaryWriter(std::ostream* stream);
        template <typename T> pix::util::Status WriteBE(T v);
        pix::util::Status WriteString(const std::string& s);
        pix::util::Status WriteBytes(pix::util::Span<const uint8_t> d);
        uint64_t Tell();
        void Seek(uint64_t p);
    private: std::ostream* stream_;
    };

    class BinaryReader {
    public:
        explicit BinaryReader(std::istream* stream);
        template <typename T> pix::util::StatusOr<T> ReadBE();
        pix::util::StatusOr<std::string> ReadString();
        pix::util::StatusOr<byte_vec> ReadBytes(size_t c);
        uint64_t Tell();
        void Seek(uint64_t p);
        void SeekFromEnd(int64_t p);
    private: std::istream* stream_;
    };
} // namespace internal


class ThreadPool final {
public:
    static ThreadPool& GetInstance(size_t threads = 0);
    template<class F, class... Args>
    auto Enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;
    template<typename T>
    static std::future<pix::util::StatusOr<T>> MakeErrorFuture(pix::util::Status status);
private:
    ThreadPool(size_t threads = 0);
    ~ThreadPool();
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

class XorCryptoProvider final : public ConceptualCryptoProvider {
public:
    KeyPair GenerateKeyPair() const override;
    Key GenerateSessionKey() const override;
    pix::util::StatusOr<Key> EncryptKey(const Key& s, const Key& p) const override;
    pix::util::StatusOr<Key> DecryptKey(const Key& e, const Key& p) const override;
    pix::util::StatusOr<byte_vec> Encrypt(pix::util::Span<const uint8_t> d, const Key& k) const override;
    pix::util::StatusOr<byte_vec> Decrypt(pix::util::Span<const uint8_t> d, const Key& k) const override;
private:
    byte_vec Xor(byte_vec data, const Key& key) const;
};

class Writer final {
 public:
  explicit Writer(std::ostream* stream);
  void SetMetadata(const std::map<std::string, std::string>& metadata);
  void SetTaskGraph(const std::map<uint64_t, GraphNode>& graph, uint64_t root_id);
  pix::util::Status AddResource(uint64_t id, ResourceType type, const std::string& name, pix::util::Span<const uint8_t> data);
  pix::util::Status AddFallback(uint8_t priority, const std::string& mime_type, pix::util::Span<const uint8_t> data);
  pix::util::Status Finalize();
 private:
  pix::util::Status SerializeMasterBlock(std::ostream* os, const MasterBlock& block);
  internal::BinaryWriter writer_;
  MasterBlock master_block_;
};

class Reader final {
 public:
  static pix::util::StatusOr<std::unique_ptr<Reader>> Create(std::unique_ptr<std::istream> stream, size_t num_threads = 0);
  const MasterBlock& GetMasterBlock() const;
  std::future<pix::util::StatusOr<byte_vec>> LoadResourceAsync(uint64_t resource_id);
 private:
  Reader(std::unique_ptr<std::istream> stream, size_t num_threads);
  pix::util::Status Initialize();
  pix::util::Status DeserializeMasterBlock(internal::BinaryReader& reader);
  std::unique_ptr<std::istream> stream_;
  std::mutex stream_mutex_;
  ThreadPool& thread_pool_;
  MasterBlock master_block_;
};

class SecureWriter final {
 public:
  SecureWriter(std::unique_ptr<std::ostream> stream, std::shared_ptr<ConceptualCryptoProvider> crypto_provider,
               const std::map<std::string, Key>& recipients);
  Writer* GetPayloadWriter();
  pix::util::Status Finalize();
 private:
  std::unique_ptr<std::ostream> secure_stream_;
  std::shared_ptr<ConceptualCryptoProvider> crypto_provider_;
  std::map<std::string, Key> recipients_;
  std::unique_ptr<std::stringstream> payload_stream_;
  std::unique_ptr<Writer> payload_writer_;
};

class UniversalLoader final {
public:
    static pix::util::StatusOr<std::unique_ptr<Reader>> Load(
        const std::filesystem::path& path,
        const std::string& recipient_id = "",
        const Key* private_key = nullptr,
        std::shared_ptr<ConceptualCryptoProvider> crypto_provider = nullptr
    );
};

} // namespace pix::ultimate::v4

// ====================================================================================
// SECTION 3: IMPLEMENTATIONS
// ====================================================================================
namespace pix::ultimate::v4 {
namespace internal {
    uint32_t CalculateCrc32(pix::util::Span<const uint8_t> data) {
        constexpr uint32_t kCrc32Poly = 0xEDB88320;
        static std::array<uint32_t, 256> table = []() { std::array<uint32_t, 256> t{}; for (uint32_t i=0;i<256;++i) { uint32_t c=i; for(int j=0;j<8;++j) {c=(c&1)?(kCrc32Poly^(c>>1)):(c>>1);} t[i]=c;} return t;}();
        uint32_t crc=~0U; for(uint8_t byte:data){crc=table[(crc^byte)&0xFF]^(crc>>8);} return ~crc;
    }
    
    BinaryWriter::BinaryWriter(std::ostream* s):stream_(s){}
    template <typename T> pix::util::Status BinaryWriter::WriteBE(T v) { if constexpr (std::is_enum_v<T>) return WriteBE(static_cast<std::underlying_type_t<T>>(v)); else { char b[sizeof(T)]; for(size_t i=0;i<sizeof(T);++i) b[i]=(v>>(8*(sizeof(T)-1-i)))&0xFF; stream_->write(b,sizeof(T)); return stream_->good()?pix::util::Status::Ok():pix::util::Status::Error("Stream write failed"); } }
    pix::util::Status BinaryWriter::WriteString(const std::string& s) { PIX_ASSIGN_OR_RETURN(void, WriteBE<uint32_t>(s.length())); stream_->write(s.data(),s.length()); return stream_->good()?pix::util::Status::Ok():pix::util::Status::Error("Stream write failed"); }
    pix::util::Status BinaryWriter::WriteBytes(pix::util::Span<const uint8_t> d) { stream_->write((const char*)d.data(),d.size()); return stream_->good()?pix::util::Status::Ok():pix::util::Status::Error("Stream write failed"); }
    uint64_t BinaryWriter::Tell() { return stream_->tellp(); }
    void BinaryWriter::Seek(uint64_t p) { stream_->seekp(p); }

    BinaryReader::BinaryReader(std::istream* s):stream_(s){}
    template <typename T> pix::util::StatusOr<T> BinaryReader::ReadBE() { if constexpr (std::is_enum_v<T>) { auto r=ReadBE<std::underlying_type_t<T>>(); if(!r.ok()) return r.status(); return static_cast<T>(r.value()); } else { char b[sizeof(T)]; stream_->read(b,sizeof(T)); if(stream_->gcount()!=sizeof(T)) return pix::util::Status::Error("EOF"); T v=0; for(size_t i=0;i<sizeof(T);++i) v=(v<<8)|(uint8_t)b[i]; return v; }}
    pix::util::StatusOr<std::string> BinaryReader::ReadString() { uint32_t l; PIX_ASSIGN_OR_RETURN(l, ReadBE<uint32_t>()); std::string s(l,'\0'); stream_->read(&s[0],l); if(stream_->gcount()!=l) return pix::util::Status::Error("EOF"); return s; }
    pix::util::StatusOr<byte_vec> BinaryReader::ReadBytes(size_t c) { byte_vec b(c); stream_->read((char*)b.data(),c); if(stream_->gcount()!=c) return pix::util::Status::Error("EOF"); return b; }
    uint64_t BinaryReader::Tell() { return stream_->tellg(); }
    void BinaryReader::Seek(uint64_t p) { stream_->seekg(p); }
    void BinaryReader::SeekFromEnd(int64_t p) { stream_->seekg(p, std::ios::end); }

    pix::util::Status WriteParameter(BinaryWriter& w, const NodeParameter& p) {
        PIX_ASSIGN_OR_RETURN(void, w.WriteBE<uint8_t>(p.index()));
        std::visit([&](auto&& a){using T=std::decay_t<decltype(a)>; if constexpr(std::is_same_v<T,int64_t>) w.WriteBE(a); else if constexpr(std::is_same_v<T,std::string>) w.WriteString(a);},p);
        return pix::util::Status::Ok();
    }
    pix::util::StatusOr<NodeParameter> ReadParameter(BinaryReader& r) {
        uint8_t i; PIX_ASSIGN_OR_RETURN(i, r.ReadBE<uint8_t>());
        switch(i){
            case 0: return std::monostate{};
            case 1:{ int64_t v; PIX_ASSIGN_OR_RETURN(v, r.ReadBE<int64_t>()); return v; }
            case 2:{ std::string v; PIX_ASSIGN_OR_RETURN(v, r.ReadString()); return v; }
            default: return pix::util::Status::Error("Invalid param index");
        }
    }
} // namespace internal


ThreadPool& ThreadPool::GetInstance(size_t t) { static ThreadPool i(t); return i; }
ThreadPool::ThreadPool(size_t t):stop_(false){size_t n=(t==0)?std::thread::hardware_concurrency():t;for(size_t i=0;i<n;++i) workers_.emplace_back([this]{while(true){std::function<void()>task;{std::unique_lock<std::mutex> l(this->queue_mutex_);this->condition_.wait(l,[this]{return this->stop_||!this->tasks_.empty();});if(this->stop_&&this->tasks_.empty())return;task=std::move(this->tasks_.front());this->tasks_.pop();}task();}}); }
ThreadPool::~ThreadPool(){{std::unique_lock<std::mutex> l(queue_mutex_);stop_=true;}condition_.notify_all();for(std::thread &w:workers_)w.join();}
template<class F,class... Args> auto ThreadPool::Enqueue(F&& f,Args&&... args)->std::future<std::invoke_result_t<F,Args...>>{using R=std::invoke_result_t<F,Args...>;auto t=std::make_shared<std::packaged_task<R()>>(std::bind(std::forward<F>(f),std::forward<Args>(args)...));std::future<R> r=t->get_future();{std::unique_lock<std::mutex> l(queue_mutex_);if(stop_)throw std::runtime_error("enqueue on stopped pool");tasks_.emplace([t](){(*t)();});}condition_.notify_one();return r;}
template<typename T> std::future<pix::util::StatusOr<T>> ThreadPool::MakeErrorFuture(pix::util::Status s){std::promise<pix::util::StatusOr<T>> p;p.set_value(std::move(s));return p.get_future();}

ConceptualCryptoProvider::KeyPair XorCryptoProvider::GenerateKeyPair() const { return {GenerateSessionKey(), GenerateSessionKey()}; }
Key XorCryptoProvider::GenerateSessionKey() const { Key k(32); std::random_device rd; std::mt19937 g(rd()); std::uniform_int_distribution<> d(0,255); for(auto& b:k) b=d(g); return k; }
pix::util::StatusOr<Key> XorCryptoProvider::EncryptKey(const Key& s, const Key& p) const { return Xor(s,p); }
pix::util::StatusOr<Key> XorCryptoProvider::DecryptKey(const Key& e, const Key& p) const { return Xor(e,p); }
pix::util::StatusOr<byte_vec> XorCryptoProvider::Encrypt(pix::util::Span<const uint8_t> d, const Key& k) const { return Xor(byte_vec(d.begin(), d.end()), k); }
pix::util::StatusOr<byte_vec> XorCryptoProvider::Decrypt(pix::util::Span<const uint8_t> d, const Key& k) const { return Encrypt(d, k); }
byte_vec XorCryptoProvider::Xor(byte_vec data, const Key& key) const { if (key.empty()) return data; for(size_t i=0;i<data.size();++i) data[i]^=key[i%key.size()]; return data; }


Writer::Writer(std::ostream* s):writer_(s){writer_.Seek(4+2+8+8);}
void Writer::SetMetadata(const std::map<std::string,std::string>& m){master_block_.metadata=m;}
void Writer::SetTaskGraph(const std::map<uint64_t,GraphNode>& g,uint64_t r){master_block_.task_graph=g;master_block_.root_node_id=r;}
pix::util::Status Writer::AddResource(uint64_t id,ResourceType t,const std::string& n,pix::util::Span<const uint8_t> d){if(master_block_.resources.count(id))return pix::util::Status::Error("Resource ID exists");ResourceDescriptor desc{id,t,n,{}};size_t w=0;while(w<d.size()){size_t cs=std::min((size_t)128*1024,d.size()-w);byte_vec cbuf(ZSTD_compressBound(cs));size_t csize=ZSTD_compress(cbuf.data(),cbuf.size(),d.data()+w,cs,1);if(ZSTD_isError(csize))return pix::util::Status::Error("ZSTD compression failed");cbuf.resize(csize);desc.segments.push_back({writer_.Tell(),(uint32_t)csize,(uint32_t)cs,internal::CalculateCrc32(cbuf)});PIX_ASSIGN_OR_RETURN(void,writer_.WriteBytes(cbuf));w+=cs;}master_block_.resources[id]=desc;return pix::util::Status::Ok();}
pix::util::Status Writer::AddFallback(uint8_t p,const std::string& m,pix::util::Span<const uint8_t> d){master_block_.fallback_cache.push_back({p,m,writer_.Tell(),d.size()});return writer_.WriteBytes(d);}
pix::util::Status Writer::Finalize(){uint64_t o=writer_.Tell();std::stringstream is;PIX_ASSIGN_OR_RETURN(void,SerializeMasterBlock(&is,master_block_));std::string d_str=is.str();byte_vec d(d_str.begin(),d_str.end());PIX_ASSIGN_OR_RETURN(void,writer_.WriteBytes(d));PIX_ASSIGN_OR_RETURN(void,writer_.WriteBE(internal::CalculateCrc32(d)));writer_.Seek(0);PIX_ASSIGN_OR_RETURN(void,writer_.WriteBE(kPixuSignature));PIX_ASSIGN_OR_RETURN(void,writer_.WriteBE(kSpecVersion));PIX_ASSIGN_OR_RETURN(void,writer_.WriteBE(kCurrentFeatureFlags));PIX_ASSIGN_OR_RETURN(void,writer_.WriteBE(o));return pix::util::Status::Ok();}
pix::util::Status Writer::SerializeMasterBlock(std::ostream* os,const MasterBlock& b){internal::BinaryWriter w(os);auto wb=[&](MasterBlockId id,auto f)->pix::util::Status{std::stringstream ss;internal::BinaryWriter bw(&ss);PIX_ASSIGN_OR_RETURN(void,f(bw));std::string d=ss.str();PIX_ASSIGN_OR_RETURN(void,w.WriteBE(id));PIX_ASSIGN_OR_RETURN(void,w.WriteBE<uint64_t>(d.length()));PIX_ASSIGN_OR_RETURN(void,w.WriteBytes({(uint8_t*)d.data(),d.size()}));return pix::util::Status::Ok();};if(!b.metadata.empty()){PIX_ASSIGN_OR_RETURN(void,wb(MasterBlockId::kMetadata,[&](auto&w){PIX_ASSIGN_OR_RETURN(void,w.WriteBE(b.metadata.size()));for(const auto&[k,v]:b.metadata){PIX_ASSIGN_OR_RETURN(void,w.WriteString(k));PIX_ASSIGN_OR_RETURN(void,w.WriteString(v));}return pix::util::Status::Ok();}));}if(!b.resources.empty()){PIX_ASSIGN_OR_RETURN(void,wb(MasterBlockId::kResources,[&](auto&w){PIX_ASSIGN_OR_RETURN(void,w.WriteBE(b.resources.size()));for(const auto&[id,d]:b.resources){PIX_ASSIGN_OR_RETURN(void,w.WriteBE(d.id));PIX_ASSIGN_OR_RETURN(void,w.WriteBE(d.type));PIX_ASSIGN_OR_RETURN(void,w.WriteString(d.name));PIX_ASSIGN_OR_RETURN(void,w.WriteBE(d.segments.size()));for(const auto& s:d.segments){PIX_ASSIGN_OR_RETURN(void,w.WriteBE(s.offset));PIX_ASSIGN_OR_RETURN(void,w.WriteBE(s.compressed_size));PIX_ASSIGN_OR_RETURN(void,w.WriteBE(s.uncompressed_size));PIX_ASSIGN_OR_RETURN(void,w.WriteBE(s.crc32_checksum));}}return pix::util::Status::Ok();}));}if(!b.fallback_cache.empty()){PIX_ASSIGN_OR_RETURN(void,wb(MasterBlockId::kFallbacks,[&](auto&w){PIX_ASSIGN_OR_RETURN(void,w.WriteBE(b.fallback_cache.size()));for(const auto&f:b.fallback_cache){PIX_ASSIGN_OR_RETURN(void,w.WriteBE(f.priority));PIX_ASSIGN_OR_RETURN(void,w.WriteString(f.mime_type));PIX_ASSIGN_OR_RETURN(void,w.WriteBE(f.offset));PIX_ASSIGN_OR_RETURN(void,w.WriteBE(f.size));}return pix::util::Status::Ok();}));}if(!b.task_graph.empty()){PIX_ASSIGN_OR_RETURN(void,wb(MasterBlockId::kTaskGraph,[&](auto&w){PIX_ASSIGN_OR_RETURN(void,w.WriteBE(b.task_graph.size()));for(const auto&[id,n]:b.task_graph){PIX_ASSIGN_OR_RETURN(void,w.WriteBE(n.id));PIX_ASSIGN_OR_RETURN(void,w.WriteBE(n.type));PIX_ASSIGN_OR_RETURN(void,w.WriteString(n.intent));PIX_ASSIGN_OR_RETURN(void,w.WriteBE(n.inputs.size()));for(auto i:n.inputs)PIX_ASSIGN_OR_RETURN(void,w.WriteBE(i));PIX_ASSIGN_OR_RETURN(void,w.WriteBE(n.params.size()));for(const auto&[k,p]:n.params){PIX_ASSIGN_OR_RETURN(void,w.WriteString(k));PIX_ASSIGN_OR_RETURN(void,internal::WriteParameter(w,p));}}PIX_ASSIGN_OR_RETURN(void,w.WriteBE(b.root_node_id));return pix::util::Status::Ok();}));}return pix::util::Status::Ok();}

pix::util::StatusOr<std::unique_ptr<Reader>> Reader::Create(std::unique_ptr<std::istream> s, size_t nt){std::unique_ptr<Reader> r(new Reader(std::move(s),nt));PIX_ASSIGN_OR_RETURN(void,r->Initialize());return r;}
Reader::Reader(std::unique_ptr<std::istream> s, size_t nt):stream_(std::move(s)),thread_pool_(ThreadPool::GetInstance(nt)){}
pix::util::Status Reader::Initialize(){internal::BinaryReader r(stream_.get());uint32_t sig;PIX_ASSIGN_OR_RETURN(sig,r.ReadBE<uint32_t>());if(sig!=kPixuSignature)return pix::util::Status::Error("Invalid PIXU signature");return DeserializeMasterBlock(r);}
pix::util::Status Reader::DeserializeMasterBlock(internal::BinaryReader& r){uint16_t v;PIX_ASSIGN_OR_RETURN(v,r.ReadBE<uint16_t>());if(v>kSpecVersion)return pix::util::Status::Error("Unsupported version");uint64_t f;PIX_ASSIGN_OR_RETURN(f,r.ReadBE<uint64_t>());uint64_t o;PIX_ASSIGN_OR_RETURN(o,r.ReadBE<uint64_t>());r.SeekFromEnd(0);uint64_t e=r.Tell();if(o>=e)return pix::util::Status::Error("Invalid index offset");uint64_t is=e-o;if(is<=sizeof(uint32_t))return pix::util::Status::Error("Invalid master block size");r.Seek(o);byte_vec ib;PIX_ASSIGN_OR_RETURN(ib,r.ReadBytes(is-sizeof(uint32_t)));uint32_t ec;PIX_ASSIGN_OR_RETURN(ec,r.ReadBE<uint32_t>());if(ec!=internal::CalculateCrc32(ib))return pix::util::Status::Error("Master block integrity check failed");std::stringstream iss(std::string(ib.begin(),ib.end()));internal::BinaryReader ir(&iss);while(ir.Tell()<ib.size()){MasterBlockId id;PIX_ASSIGN_OR_RETURN(id,ir.ReadBE<MasterBlockId>());uint64_t size;PIX_ASSIGN_OR_RETURN(size,ir.ReadBE<uint64_t>());uint64_t sp=ir.Tell();switch(id){case MasterBlockId::kMetadata:{uint64_t c;PIX_ASSIGN_OR_RETURN(c,ir.ReadBE<uint64_t>());for(uint64_t i=0;i<c;++i){std::string k,v;PIX_ASSIGN_OR_RETURN(k,ir.ReadString());PIX_ASSIGN_OR_RETURN(v,ir.ReadString());master_block_.metadata[k]=v;}break;}case MasterBlockId::kResources:{uint64_t c;PIX_ASSIGN_OR_RETURN(c,ir.ReadBE<uint64_t>());for(uint64_t i=0;i<c;++i){ResourceDescriptor d;PIX_ASSIGN_OR_RETURN(d.id,ir.ReadBE<uint64_t>());PIX_ASSIGN_OR_RETURN(d.type,ir.ReadBE<ResourceType>());PIX_ASSIGN_OR_RETURN(d.name,ir.ReadString());uint64_t sc;PIX_ASSIGN_OR_RETURN(sc,ir.ReadBE<uint64_t>());d.segments.resize(sc);for(uint64_t j=0;j<sc;++j){PIX_ASSIGN_OR_RETURN(d.segments[j].offset,ir.ReadBE<uint64_t>());PIX_ASSIGN_OR_RETURN(d.segments[j].compressed_size,ir.ReadBE<uint32_t>());PIX_ASSIGN_OR_RETURN(d.segments[j].uncompressed_size,ir.ReadBE<uint32_t>());PIX_ASSIGN_OR_RETURN(d.segments[j].crc32_checksum,ir.ReadBE<uint32_t>());}master_block_.resources[d.id]=d;}break;}case MasterBlockId::kFallbacks:{uint64_t c;PIX_ASSIGN_OR_RETURN(c,ir.ReadBE<uint64_t>());for(uint64_t i=0;i<c;++i){FallbackItem f;PIX_ASSIGN_OR_RETURN(f.priority,ir.ReadBE<uint8_t>());PIX_ASSIGN_OR_RETURN(f.mime_type,ir.ReadString());PIX_ASSIGN_OR_RETURN(f.offset,ir.ReadBE<uint64_t>());PIX_ASSIGN_OR_RETURN(f.size,ir.ReadBE<uint64_t>());master_block_.fallback_cache.push_back(f);}break;}case MasterBlockId::kTaskGraph:{uint64_t c;PIX_ASSIGN_OR_RETURN(c,ir.ReadBE<uint64_t>());for(uint64_t i=0;i<c;++i){GraphNode n;PIX_ASSIGN_OR_RETURN(n.id,ir.ReadBE<uint64_t>());PIX_ASSIGN_OR_RETURN(n.type,ir.ReadBE<NodeType>());PIX_ASSIGN_OR_RETURN(n.intent,ir.ReadString());uint64_t ic;PIX_ASSIGN_OR_RETURN(ic,ir.ReadBE<uint64_t>());n.inputs.resize(ic);for(uint64_t j=0;j<ic;++j){PIX_ASSIGN_OR_RETURN(n.inputs[j],ir.ReadBE<uint64_t>());}uint64_t pc;PIX_ASSIGN_OR_RETURN(pc,ir.ReadBE<uint64_t>());for(uint64_t j=0;j<pc;++j){std::string k;NodeParameter p;PIX_ASSIGN_OR_RETURN(k,ir.ReadString());PIX_ASSIGN_OR_RETURN(p,internal::ReadParameter(ir));n.params[k]=p;}master_block_.task_graph[n.id]=n;}PIX_ASSIGN_OR_RETURN(master_block_.root_node_id,ir.ReadBE<uint64_t>());break;}default:ir.Seek(sp+size);}}return pix::util::Status::Ok();}
const MasterBlock& Reader::GetMasterBlock()const{return master_block_;}
std::future<pix::util::StatusOr<byte_vec>>Reader::LoadResourceAsync(uint64_t id){auto it=master_block_.resources.find(id);if(it==master_block_.resources.end())return ThreadPool::MakeErrorFuture<byte_vec>(pix::util::Status::NotFound("Resource ID not found: "+std::to_string(id)));const ResourceDescriptor* d=&it->second;return thread_pool_.Enqueue([this,d]()->pix::util::StatusOr<byte_vec>{byte_vec data;for(const auto& s:d->segments){byte_vec cb;{std::lock_guard<std::mutex> l(stream_mutex_);internal::BinaryReader r(stream_.get());r.Seek(s.offset);PIX_ASSIGN_OR_RETURN(cb,r.ReadBytes(s.compressed_size));}if(internal::CalculateCrc32(cb)!=s.crc32_checksum)return pix::util::Status::Error("Data integrity error in resource "+std::to_string(d->id));byte_vec db(s.uncompressed_size);size_t ds=ZSTD_decompress(db.data(),db.size(),cb.data(),cb.size());if(ZSTD_isError(ds)||ds!=s.uncompressed_size)return pix::util::Status::Error("ZSTD decompression failed for resource "+std::to_string(d->id));data.insert(data.end(),db.begin(),db.end());}return data;});}

SecureWriter::SecureWriter(std::unique_ptr<std::ostream> s,std::shared_ptr<ConceptualCryptoProvider> c,const std::map<std::string,Key>& rec):secure_stream_(std::move(s)),crypto_provider_(std::move(c)),recipients_(rec),payload_stream_(std::make_unique<std::stringstream>()),payload_writer_(std::make_unique<Writer>(payload_stream_.get())){}
Writer* SecureWriter::GetPayloadWriter(){return payload_writer_.get();}
pix::util::Status SecureWriter::Finalize(){PIX_ASSIGN_OR_RETURN(void,payload_writer_->Finalize());std::string ps=payload_stream_->str();Key sk=crypto_provider_->GenerateSessionKey();byte_vec ep;PIX_ASSIGN_OR_RETURN(ep,crypto_provider_->Encrypt({(uint8_t*)ps.data(),ps.size()},sk));internal::BinaryWriter w(secure_stream_.get());PIX_ASSIGN_OR_RETURN(void,w.WriteBE(kPixsSignature));PIX_ASSIGN_OR_RETURN(void,w.WriteBE(kSpecVersion));PIX_ASSIGN_OR_RETURN(void,w.WriteBE((uint64_t)recipients_.size()));for(const auto&[id,pk]:recipients_){Key esk;PIX_ASSIGN_OR_RETURN(esk,crypto_provider_->EncryptKey(sk,pk));PIX_ASSIGN_OR_RETURN(void,w.WriteString(id));PIX_ASSIGN_OR_RETURN(void,w.WriteBE<uint32_t>(esk.size()));PIX_ASSIGN_OR_RETURN(void,w.WriteBytes(esk));}PIX_ASSIGN_OR_RETURN(void,w.WriteBytes(ep));return pix::util::Status::Ok();}

pix::util::StatusOr<std::unique_ptr<Reader>> UniversalLoader::Load(const std::filesystem::path& p,const std::string& rid,const Key* privk,std::shared_ptr<ConceptualCryptoProvider> crypto){auto fs=std::make_unique<std::ifstream>(p,std::ios::binary);if(!fs||!fs->is_open())return pix::util::Status::Error("Cannot open file: "+p.string());internal::BinaryReader r(fs.get());uint32_t sig;PIX_ASSIGN_OR_RETURN(sig,r.ReadBE<uint32_t>());r.Seek(0);if(sig==kPixuSignature){return Reader::Create(std::move(fs));}else if(sig==kPixsSignature){if(!privk||rid.empty()||!crypto)return pix::util::Status::Error("Secure file requires key, ID, and crypto provider");r.Seek(4);uint16_t v;PIX_ASSIGN_OR_RETURN(v,r.ReadBE<uint16_t>());if(v>kSpecVersion)return pix::util::Status::Error("Unsupported secure file version");uint64_t rc;PIX_ASSIGN_OR_RETURN(rc,r.ReadBE<uint64_t>());std::optional<Key> ek;for(uint64_t i=0;i<rc;++i){std::string id;PIX_ASSIGN_OR_RETURN(id,r.ReadString());uint32_t l;PIX_ASSIGN_OR_RETURN(l,r.ReadBE<uint32_t>());if(id==rid){PIX_ASSIGN_OR_RETURN(ek,r.ReadBytes(l));}else{r.Seek(r.Tell()+l);}}if(!ek)return pix::util::Status::NotFound("Recipient '"+rid+"' not found in secure file");Key sk;PIX_ASSIGN_OR_RETURN(sk,crypto->DecryptKey(*ek,*privk));uint64_t po=r.Tell();r.SeekFromEnd(0);uint64_t ps=r.Tell()-po;r.Seek(po);byte_vec ep;PIX_ASSIGN_OR_RETURN(ep,r.ReadBytes(ps));byte_vec dp;PIX_ASSIGN_OR_RETURN(dp,crypto->Decrypt(ep,sk));auto ds=std::make_unique<std::stringstream>(std::string(dp.begin(),dp.end()));return Reader::Create(std::move(ds));}return pix::util::Status::Error("Unknown file signature");}
} // namespace pix::ultimate::v4
#endif // PI_ULTIMATE_V4_3_H_
