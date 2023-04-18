#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
void myInference()
{
}
void myTrain()
{

}
void testSelfFunctionCuda()
{
   std::string model_path = "vae.pt";

    // Load the serialized model
    torch::jit::script::Module module = torch::jit::load(model_path);
     torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        std::cout<<"now using cuda"<<std::endl;
        module.to(device); // move the module to the device
    }

     torch::jit::Method myMethod= module.get_method("testFunc");
    //torch::jit::Function myFun = myMethod.function();
     
    // Print the output
    //run myFunc
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0};
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor in1 = torch::from_blob(data.data(), {1, 4},options);
    auto scalarTensor = torch::tensor(0.5, options).to(torch::kCUDA);
     torch::jit::Stack stack;
     in1=in1.to(torch::kCUDA);
    // Push the input tensor onto the stack
    stack.push_back(in1);
    stack.push_back(scalarTensor);
// Execute the function
     //myMethod.run(stack);

/*Get the output tensor from the stack, if only one return:
 torch::Tensor output = std::move(stack.front().toTensor());
 */
// Here is how to deal with multiple returns
auto outElements=myMethod(stack).toTuple()->elements();
at::Tensor output=outElements[0].toTensor();
float ru2=outElements[1].to<float>();
// Call myFun with the input data
    std::cout << "TestFunc Output: " << output <<ru2<< std::endl;
}

void testSelfFunctionNormal()
{
   std::string model_path = "vae.pt";

    // Load the serialized model
    torch::jit::script::Module module = torch::jit::load(model_path);
  
     torch::jit::Method myMethod= module.get_method("testFunc");
    //torch::jit::Function myFun = myMethod.function();
     
    // Print the output
    //run myFunc
    std::vector<float> data = {1.0, 2.0, 3.0, 4.0};
    torch::Tensor in1 = torch::from_blob(data.data(), {1, 4}, torch::kFloat32);
     torch::jit::Stack stack;
    // Push the input tensor onto the stack
    stack.push_back(in1);
    stack.push_back(0.5);
// Execute the function
     //myMethod.run(stack);

/*Get the output tensor from the stack, if only one return:
 torch::Tensor output = std::move(stack.front().toTensor());
 */
// Here is how to deal with multiple returns
auto outElements=myMethod(stack).toTuple()->elements();
at::Tensor output=outElements[0].toTensor();
float ru2=outElements[1].to<float>();
// Call myFun with the input data
    std::cout << "TestFunc Output: " << output <<ru2<< std::endl;
}
int main() {
  std::string model_path = "vae.pt";

    // Load the serialized model
    torch::jit::script::Module module = torch::jit::load(model_path);
     //return 0;
    // Generate some input data (an arbitrary vector X)
    std::vector<float> vc = {1,2,3,4,5,6,7,8,9,10};

    // Convert the input data to a tensor
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    //torch::Tensor input_tensor = torch::from_blob(input_data.data(), {1, input_data.size()}, options);
     auto tensor2 = torch::from_blob(vc.data(), {int64_t(vc.size())}, opts).clone();
     //tensor2=tensor2.reshape({1,int64_t(vc.size())});
    // Perform inference
    at::Tensor output_tensor = module.forward({tensor2}).toTuple()->elements()[0].toTensor();

    // Print the output
    std::cout << "Output: " << output_tensor << std::endl;
    // run the mytrain function
    testSelfFunctionNormal();
    return 0;
}
