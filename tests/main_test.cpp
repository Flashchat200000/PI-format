#include <gtest/gtest.h>
#include "pi/pi.hpp"

using namespace pix::ultimate::v4;

TEST(PixFormatTest, BasicRoundTrip) {
    const std::string test_file = "round_trip_test.pixu";
    MasterBlock original_block;
    original_block.metadata = {{"test_key", "test_value"}};
    original_block.resources[101] = {101, ResourceType::kMeshGeometry, "test_mesh", {}};

    {
        auto file_stream = std::make_unique<std::ofstream>(test_file, std::ios::binary);
        ASSERT_TRUE(file_stream && file_stream->is_open());

        Writer writer(file_stream.get());
        writer.SetMetadata(original_block.metadata);
        auto status = writer.AddResource(101, ResourceType::kMeshGeometry, "test_mesh", {'d','a','t','a'});
        ASSERT_TRUE(status.ok());
        status = writer.Finalize();
        ASSERT_TRUE(status.ok());
    }

    auto reader_or = UniversalLoader::Load(test_file);
    ASSERT_TRUE(reader_or.ok());

    const auto& read_block = reader_or.value()->GetMasterBlock();

    ASSERT_EQ(read_block.metadata.size(), 1);
    ASSERT_EQ(read_block.metadata.at("test_key"), "test_value");
    ASSERT_EQ(read_block.resources.size(), 1);
    ASSERT_TRUE(read_block.resources.count(101));
    ASSERT_EQ(read_block.resources.at(101).name, "test_mesh");
}
