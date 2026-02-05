#include <inference.h>

// 怎么把 vector 的数据弄到 GPU 上？
// 怎么告诉 TRT 我们的 Batch Size 是多少？
// 那个致命的 enqueueV2 怎么调用？
// 最后怎么把数据弄回来？
// 提示： TRT 中，输入和输出的显存地址是放在一个 void* bindings[] 数组里传给 enqueueV2 的。
bool InferenceEngine::run(const std::vector<float>& inputData, std::vector<float>& outputData) {
    // 1. 检查数据大小是否匹配
    if(inputData.size() != inputSize_) {
        return false;
    }

    // 2. 将数据从 Host 拷贝到 Device (Async)
    cudaMemcpyAsync(d_input_, inputData.data(), inputSize_ * sizeof(float), cudaMemcpyHostToDevice, stream_);

    // 3. 绑定输入输出指针
    void* bindings[2];
    bindings[inputIndex_] = d_input_;
    bindings[outputIndex_] = d_output_;

    // 4. 执行推理
    // TODO: 写出 enqueueV2 语句
    context_->enqueueV2(bindings, stream_, nullptr);


    // 确保输出容器大小足够，否则 resize
    if (outputData.size() != outputSize_) {
        outputData.resize(outputSize_);
    }
    // 5. 将结果从 Device 拷贝回 Host (Async)
    // TODO: 写出 cudaMemcpyAsync 语句
    cudaMemcpyAsync(outputData.data(), d_output_, outputSize_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);

    // 6. 同步等待
    // TODO: 写出同步语句
    cudaStreamSynchronize(stream_);

    return true;
}