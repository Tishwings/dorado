#include "read_pipeline/nodes/ReadToBamTypeNode.h"

#include "MessageSinkUtils.h"
#include "read_pipeline/base/messages/SimplexRead.h"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <vector>

#define TEST_GROUP "[ReadToBamTypeNodeTest]"
#define DEFINE_TEST(name) CATCH_TEST_CASE(TEST_GROUP " " name, TEST_GROUP)

DEFINE_TEST("ReadToBamTypeNode normalizes empty sample_id") {
    dorado::PipelineDescriptor pipeline_desc;
    std::vector<dorado::Message> messages;
    auto sink = pipeline_desc.add_node<MessageSinkToVector>({}, 100, messages);
    pipeline_desc.add_node<dorado::ReadToBamTypeNode>({sink}, false, 1, std::nullopt, 1000, 0);
    auto pipeline = dorado::Pipeline::create(std::move(pipeline_desc), nullptr);

    auto read = std::make_unique<dorado::SimplexRead>();
    read->read_common.read_id = "read_0";
    read->read_common.seq = "ACGTACGT";
    read->read_common.qstring = "********";
    pipeline->push_message(std::move(read));
    pipeline->terminate({.fast = dorado::utils::AsyncQueueTerminateFast::No});

    CATCH_REQUIRE(messages.size() == 1);
    CATCH_REQUIRE(messages.front().holds<dorado::BamMessage>());
    const auto& bam_message = messages.front().get<dorado::BamMessage>();
    CATCH_REQUIRE(bam_message.data != nullptr);
    CATCH_CHECK(bam_message.data->read_attrs.sample_id == "no_sample");
}