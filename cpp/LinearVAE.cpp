#include <LinearVAE.hpp>
#include <torch/optim/adam.h>
using namespace TROCHPACK_VAE;
using namespace std;
using namespace torch;

void LinearVAE::loadModule(std::string path)
{  cout<<"load from "+path+"\r\n";
   module = torch::jit::load(path);
   getDimensionMethod=newtorchMethod(module.get_method("getDimension"));
   getDimension();

   getMuEstimationMethod=newtorchMethod(module.get_method("getMuEstimation"));

   loadPriorDistMethod=newtorchMethod(module.get_method("loadPriorDist"));

   lossUnderNormalMethod=newtorchMethod(module.get_method("lossUnderNormal"));
   auto parameters = module.parameters();
 //create the optimiser
   vector<torch::Tensor> params_vec;
for (auto it = parameters.begin(); it != parameters.end(); ++it) {
    params_vec.push_back(*it);
}
  myOpt=newtorchOptimiser(params_vec, torch::optim::AdamOptions(0.001));
   //getDimension();
   cout<<"done"<<endl;
}
void LinearVAE::getDimension()
{  
    torch::jit::Stack stack;
    auto ru=(getDimensionMethod.get()[0])(stack).toTuple();
    auto outElements=ru->elements();
    inputDimension=outElements[0].to<uint64_t>();
    latentDimension=outElements[1].to<uint64_t>();
    //cout<<"input dimension:"<<inputDimension<<" latent dimension:"<<latentDimension<<endl;
   // cout<<outElements<<endl;
}
void LinearVAE::loadPriorDist(float pmu,float psigma,float a0,float b0)
{
     torch::jit::Stack stack;
    stack.push_back(torch::tensor({pmu}));
    stack.push_back(torch::tensor({psigma}));
    stack.push_back(torch::tensor({a0}));
    stack.push_back(torch::tensor({b0}));
    loadPriorDistMethod.get()[0](stack);
}
void LinearVAE::getMuEstimation()
{
    torch::jit::Stack stack;
    auto ru=(getMuEstimationMethod.get()[0])(stack).toTuple();
    auto outElements=ru->elements();

    //float ru2=outElements[1].to<float>();
    resultMu=outElements[0].toTensor().mean().item<float>();
    resultSigma=outElements[1].toTensor().mean().item<float>();
}
void LinearVAE::runForward(std::vector<float> data)
{
uint64_t rows=data.size()/inputDimension;
torch::Tensor in1 = torch::from_blob(data.data(), {rows, inputDimension}, torch::kFloat32);

auto outElements=module.forward({in1}).toTuple()->elements();
/*for(int i=0;i<5;i++)
{
    cout<<outElements[i].toTensor()<<endl;
}*/
getMuEstimation();
}
void LinearVAE::learnStep(std::vector<float> data)
{   torch::jit::Stack stack;
    uint64_t rows=data.size()/inputDimension;
    torch::Tensor in1 = torch::from_blob(data.data(), {rows, inputDimension}, torch::kFloat32);
    // forward
    auto outElements=module.forward({in1}).toTuple()->elements();
    auto x_recon=outElements[0].toTensor();
    auto muZ=outElements[0].toTensor();
    auto logvarZ=outElements[0].toTensor();
    auto mu=outElements[0].toTensor();
    auto logvar=outElements[0].toTensor();
    //get loss
    stack.push_back(x_recon);
    stack.push_back(in1);
    stack.push_back(mu);
    stack.push_back(logvar);
    auto ru=(lossUnderNormalMethod.get()[0])(stack).toTensor();
    resultLoss=ru.item<float>();
    myOpt->zero_grad();
    ru.backward();
    myOpt->step();
    //cout<<ru<<endl;
}
void LinearVAE::learnStep(torch::Tensor data)
{   torch::jit::Stack stack;
    auto outElements=module.forward({data}).toTuple()->elements();
    auto x_recon=outElements[0].toTensor();
    auto muZ=outElements[0].toTensor();
    auto logvarZ=outElements[0].toTensor();
    auto mu=outElements[0].toTensor();
    auto logvar=outElements[0].toTensor();
    //get loss
    stack.push_back(x_recon);
    stack.push_back(data);
    stack.push_back(mu);
    stack.push_back(logvar);
    auto ru=(lossUnderNormalMethod.get()[0])(stack).toTensor();
    resultLoss=ru.item<float>();
    myOpt->zero_grad();
    ru.backward();
    myOpt->step();
}