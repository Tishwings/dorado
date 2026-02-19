#pragma once

#include <cstdint>
#include <memory>
#include <variant>

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
class Message {
    std::variant<std::monostate,
                 SimplexReadPtr,
                 BamMessage,
                 ReadPairPtr,
                 CacheFlushMessage,
                 DuplexReadPtr,
                 CorrectionAlignmentsPtr>
            m_message;

public:
    // Constructors for each message type.
    Message(SimplexReadPtr&& message);
    Message(BamMessage&& message);
    Message(ReadPairPtr&& message);
    Message(CacheFlushMessage&& message);
    Message(DuplexReadPtr&& message);
    Message(CorrectionAlignmentsPtr&& message);

    explicit Message();
    ~Message();

    Message(const Message&) = delete;
    Message& operator=(const Message&) = delete;
    Message(Message&&) noexcept;
    Message& operator=(Message&&) noexcept;

    // See if this holds a certain message type.
    template <typename T>
    bool holds() const;

    // Get a reference to the message type.
    // Throws if the type doesn't match what's held.
    template <typename T>
    const T& get() const;

    // Extract the message that's held.
    // Throws is the type doesn't match what's held.
    template <typename T>
    T take();

    // Exposed for debugging.
    std::size_t index() const;
};

bool is_read_message(const Message& message);

ReadCommon& get_read_common_data(Message& message);
const ReadCommon& get_read_common_data(const Message& message);

// Ensures the raw_data field is non-empty, which it won't necessarily be for DuplexRead.
void materialise_read_raw_data(Message& message);

}  // namespace dorado
