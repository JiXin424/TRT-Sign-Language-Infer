#pragma once
#include <memory>
#include <vector>
#include <string>
#include <NvInfer.h>

// 使用 RAII 管理 CUDA 指针 (这是一个很好的习惯)
struct CudaDeleter {
    void operator()(void* ptr) { cudaFree(ptr); }
};
using CudaPtr = std::unique_ptr<void, CudaDeleter>;

class InferenceEngine {
public:
    // 构造函数：负责加载模型、初始化 Context
    InferenceEngine(const std::string& modelPath);
    
    // 析构函数
    ~InferenceEngine();

    // 核心推理函数
    // inputData: 主机端(CPU)的输入数据
    // outputData: 用于接收结果的容器
    bool run(const std::vector<float>& inputData, std::vector<float>& outputData);

private:
    // TensorRT 的组件
    // 注意：在实际代码中，这些指针也应该用 std::unique_ptr 管理，为了简化演示暂时用裸指针
    // 但 context 必须手动 destroy
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    
    // CUDA 流
    cudaStream_t stream_;

    // 显存指针 (输入和输出)
    void* d_input_ = nullptr; 
    void* d_output_ = nullptr;
    
    // 绑定索引 (Binding Index)
    int inputIndex_;
    int outputIndex_;
    
    // 辅助变量
    int inputSize_;  // float 个数
    int outputSize_; // float 个数
};

// 怎么把 vector 的数据弄到 GPU 上？
// 怎么告诉 TRT 我们的 Batch Size 是多少？
// 那个致命的 enqueueV2 怎么调用？
// 最后怎么把数据弄回来？
// 提示： TRT 中，输入和输出的显存地址是放在一个 void* bindings[] 数组里传给 enqueueV2 的。