#include "secondary/architectures/model_torch_base.h"

namespace dorado::secondary {

torch::Device ModelTorchBase::get_device() const {
    // Get the device of the first parameter as a representative.
    for (const auto& param : this->parameters()) {
        if (param.defined()) {
            return param.device();
        }
    }
    return torch::Device(torch::kCPU);
}

// Convert the model to half precision
void ModelTorchBase::to_half() {
    std::lock_guard<std::mutex> lock(m_mutex_write);
    this->to(torch::kHalf);
    m_half_precision = true;
}

void ModelTorchBase::set_normalise(const bool val) {
    std::lock_guard<std::mutex> lock(m_mutex_write);
    m_normalise = val;
}

// Sets the eval mode.
void ModelTorchBase::set_eval() {
    std::lock_guard<std::mutex> lock(m_mutex_write);
    this->eval();
}

void ModelTorchBase::to_device(torch::Device device) {
    std::lock_guard<std::mutex> lock(m_mutex_write);
    this->to(device);
}

// Predict on a batch with device and precision handling.
at::Tensor ModelTorchBase::predict_on_batch(BatchedData batched_data) {
    batched_data = prepare_batch_input(batched_data, false);
    at::Tensor x = predict_on_device_batch(batched_data);
    x = x.cpu();
    return x;
}

BatchedData ModelTorchBase::prepare_batch_input(BatchedData batched_data,
                                                const bool non_blocking) const {
    batched_data.features = batched_data.features.to(get_device(), non_blocking);
    if (m_half_precision) {
        batched_data.features = batched_data.features.to(torch::kHalf, non_blocking);
    }
    if (batched_data.refseqs) {
        batched_data.refseqs = {batched_data.refseqs->to(get_device(), non_blocking)};
    }
    return batched_data;
}

at::Tensor ModelTorchBase::predict_on_device_batch(const BatchedData& batched_data) {
    std::lock_guard<std::mutex> lock(m_mutex_write);
    at::Tensor x = batched_data.features;
    x = forward(std::move(x));
    if (m_half_precision) {
        x = x.to(torch::kFloat);
    }
    if (m_normalise) {
        x = torch::softmax(x, -1);
    }
    return x;
}

std::unordered_set<std::string> ModelTorchBase::get_non_persistent_buffers() const {
    std::lock_guard<std::mutex> lock(m_mutex_write);
    return m_non_persistent_buffers;
}

void ModelTorchBase::add_nonpersistent_buffer(std::string name) {
    std::lock_guard<std::mutex> lock(m_mutex_write);
    m_non_persistent_buffers.emplace(std::move(name));
}

bool ModelTorchBase::requires_ref() const { return false; }

}  // namespace dorado::secondary
