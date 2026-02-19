#pragma once

#include <cstdint>
#include <memory>
#include <variant>

// TODO: remove these by hiding Message's dtor
#include "messages/CorrectionAlignments.h"
#include "messages/DuplexRead.h"
#include "messages/ReadPair.h"
#include "messages/SimplexRead.h"
#include "utils/types.h"

namespace dorado {

class ClientInfo;
class DuplexRead;
class HtsData;
class ReadCommon;
class SimplexRead;
struct CorrectionAlignments;
struct ReadPair;

using CorrectionAlignmentsPtr = std::unique_ptr<CorrectionAlignments>;
using DuplexReadPtr = std::unique_ptr<DuplexRead>;
using ReadPairPtr = std::unique_ptr<ReadPair>;
using SimplexReadPtr = std::unique_ptr<SimplexRead>;

class CacheFlushMessage {
public:
    int32_t client_id;
};

struct BamMessage {
    std::unique_ptr<HtsData> data;
    std::shared_ptr<ClientInfo> client_info;
};

// The Message type is a std::variant that can hold different types of message objects.
// It is currently able to store:
// - a SimplexReadPtr object, which represents a single Simplex read
// - a DuplexReadPtr object, which represents a single Duplex read
// - a BamMessage object, composite class holding a BamPtr (which represents a raw BAM alignment record) and ClientInfo
// - a ReadPair object, which represents a pair of reads for duplex calling
// - a CorrectionAlignments, which holds alignment information per read to be corrected
// To add more message types, simply add them to the list of types in the std::variant.
using Message = std::variant<SimplexReadPtr,
                             BamMessage,
                             ReadPairPtr,
                             CacheFlushMessage,
                             DuplexReadPtr,
                             CorrectionAlignmentsPtr>;
// 32 was chosen arbitrarily (it's the current size). In the future we might want to change the
// logic to have |Message| be the full objects, ie not holding pointers, and instead pass around
// a |unique_ptr<Message>|.
static_assert(sizeof(Message) <= 32,
              "Messages should be kept small since they're shared by all nodes");

bool is_read_message(const Message& message);

ReadCommon& get_read_common_data(Message& message);
const ReadCommon& get_read_common_data(const Message& message);

// Ensures the raw_data field is non-empty, which it won't necessarily be for DuplexRead.
void materialise_read_raw_data(Message& message);

}  // namespace dorado
