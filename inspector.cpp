// ====================================================================================
// PIX ULTIMATE INSPECTOR (pix-inspect)
//
// A professional command-line tool to dissect and display the contents of
// .pixu and .pixs files. Supports argument parsing and formatted output.
//
// Version: 1.1 (Professional Edition)
// ====================================================================================

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <cstdlib> // For EXIT_SUCCESS, EXIT_FAILURE

// Assumes 'pix_ultimate.hpp' is in the same directory or in the include path.
#include "pix_ultimate.hpp"

// Use the library's namespace.
using namespace pix::ultimate::v4;

// --- Forward Declarations for Printing ---
void PrintUsage();
void PrintHeader(const std::string& title);
void PrintSummary(const std::filesystem::path& path, std::shared_ptr<Reader> reader);
void PrintMetadata(const MasterBlock& block);
void PrintResources(const MasterBlock& block);
void PrintFallbacks(const MasterBlock& block);
void PrintTaskGraph(const MasterBlock& block);

// --- Main Application Logic ---

int main(int argc, char* argv[]) {
    if (argc != 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        PrintUsage();
        return EXIT_SUCCESS;
    }

    std::filesystem::path file_path = argv[1];

    if (!std::filesystem::exists(file_path)) {
        std::cerr << "Error: File not found: " << file_path << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "=========================================" << std::endl;
    std::cout << " PIX ULTIMATE FILE INSPECTOR v1.1" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // For this tool, we only inspect unencrypted files.
    // A more advanced tool could accept key arguments to decrypt .pixs files.
    auto reader_or = UniversalLoader::Load(file_path);

    if (!reader_or.ok()) {
        std::cerr << "\n[ERROR] Failed to load file: " << reader_or.status().message() << std::endl;
        if (reader_or.status().code() == pix::util::StatusCode::kNotFound) {
             std::cerr << "Hint: This may be an encrypted .pixs file. This tool inspects the structure of unencrypted .pixu files." << std::endl;
        }
        return EXIT_FAILURE;
    }

    std::cout << "\nFile loaded and master block checksum VERIFIED." << std::endl;

    auto reader = reader_or.value();
    const auto& master_block = reader->GetMasterBlock();
    
    PrintSummary(file_path, reader);
    PrintMetadata(master_block);
    PrintResources(master_block);
    PrintFallbacks(master_block);
    PrintTaskGraph(master_block);

    std::cout << "\n--- Inspection Complete ---" << std::endl;

    return EXIT_SUCCESS;
}


// --- Implementation of Printing Functions ---

void PrintUsage() {
    std::cout << "PIX Ultimate Inspector (pix-inspect)" << std::endl;
    std::cout << "A tool to display the internal structure of a .pixu file." << std::endl;
    std::cout << "\nUSAGE:" << std::endl;
    std::cout << "  pix-inspect <path_to_file.pixu>" << std::endl;
    std::cout << "\nOPTIONS:" << std::endl;
    std::cout << "  -h, --help    Show this help message." << std::endl;
}

void PrintHeader(const std::string& title) {
    std::cout << "\n--- " << title << " " << std::string(40 - title.length(), '-') << std::endl;
}

void PrintSummary(const std::filesystem::path& path, std::shared_ptr<Reader> reader) {
    PrintHeader("File Summary");
    std::cout << std::left;
    std::cout << "  " << std::setw(20) << "File Path:" << path << std::endl;
    std::cout << "  " << std::setw(20) << "File Size:" << std::filesystem::file_size(path) << " bytes" << std::endl;
}

void PrintMetadata(const MasterBlock& block) {
    PrintHeader("Metadata Block");
    if (block.metadata.empty()) {
        std::cout << "  (No metadata)" << std::endl;
        return;
    }
    std::cout << std::left;
    for (const auto& [key, value] : block.metadata) {
        std::cout << "  " << std::setw(20) << (key + ":") << value << std::endl;
    }
}

void PrintResources(const MasterBlock& block) {
    PrintHeader("Resources Block (" + std::to_string(block.resources.size()) + ")");
    std::cout << std::left;
    std::cout << "  " << std::setw(8) << "ID" << std::setw(30) << "Name" << std::setw(10) << "Segments" << "Size" << std::endl;
    std::cout << "  " << std::string(70, '-') << std::endl;
    for (const auto& [id, desc] : block.resources) {
        uint64_t total_size = 0;
        for (const auto& seg : desc.segments) { total_size += seg.uncompressed_size; }
        
        std::cout << "  " << std::setw(8) << desc.id
                  << std::setw(30) << desc.name
                  << std::setw(10) << desc.segments.size()
                  << total_size << " bytes" << std::endl;
    }
}

void PrintFallbacks(const MasterBlock& block) {
    PrintHeader("Fallbacks Block (" + std::to_string(block.fallback_cache.size()) + ")");
     if (block.fallback_cache.empty()) {
        std::cout << "  (No fallbacks)" << std::endl;
        return;
    }
    std::cout << std::left;
    std::cout << "  " << std::setw(12) << "Priority" << std::setw(30) << "MIME Type" << "Size" << std::endl;
    std::cout << "  " << std::string(70, '-') << std::endl;
    for (const auto& item : block.fallback_cache) {
        std::cout << "  " << std::setw(12) << (int)item.priority
                  << std::setw(30) << item.mime_type
                  << item.size << " bytes" << std::endl;
    }
}

void PrintTaskGraph(const MasterBlock& block) {
    PrintHeader("Task Graph Block (" + std::to_string(block.task_graph.size()) + " nodes)");
    if (block.task_graph.empty()) {
        std::cout << "  (No task graph)" << std::endl;
        return;
    }
    std::cout << "  Root Node ID: " << block.root_node_id << std::endl;
    for (const auto& [id, node] : block.task_graph) {
        std::cout << "\n  [Node " << node.id << "] - " << node.intent << std::endl;
        std::cout << "    " << std::setw(12) << "Type:" << "NODE_TYPE_" << (int)node.type << std::endl;
        
        std::cout << "    " << std::setw(12) << "Inputs:";
        if (node.inputs.empty()) {
            std::cout << "(none)" << std::endl;
        } else {
            for (size_t i = 0; i < node.inputs.size(); ++i) {
                std::cout << node.inputs[i] << (i == node.inputs.size() - 1 ? "" : ", ");
            }
            std::cout << std::endl;
        }

        if(!node.params.empty()){
            std::cout << "    " << "Parameters:" << std::endl;
            for(const auto& [key, param] : node.params) {
                std::cout << "      - " << std::setw(20) << std::left << (key + ":");
                std::visit([](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, std::monostate>) std::cout << "(monostate)";
                    else if constexpr (std::is_same_v<T, int64_t>) std::cout << arg << " (int64)";
                    else if constexpr (std::is_same_v<T, std::string>) std::cout << "\"" << arg << "\" (string)";
                }, param);
                std::cout << std::endl;
            }
        }
    }
}
