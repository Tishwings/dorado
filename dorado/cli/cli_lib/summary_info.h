#pragma once

#include "hts_writer/SummaryFileWriter.h"
#include "read_pipeline/base/ReadInitialiser.h"

#include <filesystem>
#include <tuple>
#include <vector>

namespace dorado::cli {

std::tuple<hts_writer::SummaryFileWriter::FieldFlags, AlignmentCounts> make_summary_info(
        const std::vector<std::filesystem::path>& all_files);

}  // namespace dorado::cli
