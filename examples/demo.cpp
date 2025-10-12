// ====================================================================================
// PIX ULTIMATE DEMO GENERATOR (create-demo)
//
// Creates a feature-rich .pixu and a corresponding .pixs file for testing
// and demonstration of the PIX Ultimate ecosystem tools.
//
// Version: 1.1 (Professional Edition)
// ====================================================================================

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <fstream>
#include <cstdlib> // For EXIT_SUCCESS, EXIT_FAILURE

// Assumes 'pix_ultimate.hpp' is in the same directory or in the include path.
#include "pi/pi.hpp"

// Use the library's namespace.
using namespace pix::ultimate::v4;

// A robust helper for checking status results in main.
void CheckStatus(const pix::util::Status& status, const std::string& context) {
  if (!status.ok()) {
    std::cerr << "[FATAL ERROR] " << context << ": " << status.message() << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main() {
    std::cout << "--- PIX Ultimate Professional Demo Generator ---" << std::endl;

    const std::string standard_file_name = "scene.pixu";
    const std::string secure_file_name = "scene.pixs";
    
    // --- 1. Create the MasterBlock Data ---
    MasterBlock master_block;

    // A more detailed metadata block.
    master_block.metadata = {
        {"creator", "PIX Demo Generator v1.1"},
        {"scene_name", "Test Chamber 01"},
        {"quality_level", "production"}
    };

    // Multiple resources.
    byte_vec mesh_data(1024 * 32, 'M'); // 32KB Mesh
    byte_vec texture_data(1024 * 128, 'T'); // 128KB Texture
    
    // The task graph now represents a logical rendering flow.
    master_block.task_graph = {
        {1, {1, NodeType::kLoadResource, "Load Player Mesh", {}, {{"resource_id", (int64_t)101}}}},
        {2, {2, NodeType::kLoadResource, "Load Player Texture", {}, {{"resource_id", (int64_t)202}}}},
        {3, {3, NodeType::kRenderGbuffer, "Render Player G-Buffer", {1, 2}, {{"shader_quality", (int64_t)3}, {"material_name", std::string("player_ubershader")}}}},
    };
    master_block.root_node_id = 3;

    // --- 2. Write the Standard .pixu File ---
    std::cout << "\n[1/2] Writing standard file: " << standard_file_name << std::endl;
    {
        auto file_stream = std::make_unique<std::ofstream>(standard_file_name, std::ios::binary);
        if (!file_stream || !file_stream->is_open()) {
            std::cerr << "Fatal: Could not open " << standard_file_name << " for writing." << std::endl;
            return EXIT_FAILURE;
        }
        
        Writer writer(file_stream.get());
        writer.SetMetadata(master_block.metadata);
        writer.SetTaskGraph(master_block.task_graph, master_block.root_node_id);

        CheckStatus(writer.AddResource(101, ResourceType::kMeshGeometry, "player.mesh", mesh_data), "AddResource(mesh)");
        CheckStatus(writer.AddResource(202, ResourceType::kVirtualTexture, "player_albedo.texture", texture_data), "AddResource(texture)");
        CheckStatus(writer.AddFallback(1, "image/png", {'p','n','g',' ','d','a','t','a'}), "AddFallback");
        
        CheckStatus(writer.Finalize(), "Finalize(standard_file)");
        std::cout << "  Success." << std::endl;
    }

    // --- 3. Write the Secure .pixs File ---
    std::cout << "\n[2/2] Writing secure file: " << secure_file_name << std::endl;
    {
        auto crypto = std::make_shared<XorCryptoProvider>();
        auto bobs_keys = crypto->GenerateKeyPair();
        std::map<std::string, Key> recipients = {
            {"bob", bobs_keys.public_key},
            {"alice", crypto->GenerateKeyPair().public_key}
        };

        auto file_stream = std::make_unique<std::ofstream>(secure_file_name, std::ios::binary);
         if (!file_stream || !file_stream->is_open()) {
            std::cerr << "Fatal: Could not open " << secure_file_name << " for writing." << std::endl;
            return EXIT_FAILURE;
        }

        SecureWriter writer(std::move(file_stream), crypto, recipients);
        
        // Use the same data, but could be different for a secure payload.
        writer.GetPayloadWriter()->SetMetadata(master_block.metadata);
        writer.GetPayloadWriter()->SetTaskGraph(master_block.task_graph, master_block.root_node_id);
        CheckStatus(writer.GetPayloadWriter()->AddResource(101, ResourceType::kMeshGeometry, "player.mesh", mesh_data), "Secure::AddResource(mesh)");
        CheckStatus(writer.GetPayloadWriter()->AddResource(202, ResourceType::kVirtualTexture, "player_albedo.texture", texture_data), "Secure::AddResource(texture)");

        CheckStatus(writer.Finalize(), "Finalize(secure_file)");
        std::cout << "  Success. Encrypted for 'bob' and 'alice'." << std::endl;
    }

    std::cout << "\n--- Demo File Generation Complete ---" << std::endl;
    return EXIT_SUCCESS;
}
