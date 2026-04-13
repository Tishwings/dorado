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
torch::Tensor ModelTorchBase::predict_on_batch(torch::Tensor x) {
    x = prepare_batch_input(std::move(x), false);
    x = predict_on_device_batch(std::move(x));
    x = x.cpu();
    return x;
}

torch::Tensor ModelTorchBase::prepare_batch_input(torch::Tensor x, const bool non_blocking) const {
    x = x.to(get_device(), non_blocking);
    if (m_half_precision) {
        x = x.to(torch::kHalf, non_blocking);
    }
    return x;
}

torch::Tensor ModelTorchBase::predict_on_device_batch(torch::Tensor x) {
    std::lock_guard<std::mutex> lock(m_mutex_write);
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

}  // namespace dorado::secondary
