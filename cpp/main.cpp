#include <LinearVAE.hpp>
#include <vector>
using namespace std;
using namespace TROCHPACK_VAE;
int main()
{
    LinearVAE lv;
    lv.loadModule("linearVAE.pt");
    std::vector<float> vc = {6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.4,6.2};
    lv.runForward(vc);
    std::cout<<"mu="<<lv.resultMu<<",sigma="<<lv.resultSigma<<endl;
    int num_samples = 1000;
    int input_dim = 10;
    auto noiseX = torch::randn({num_samples, input_dim});
    auto baseX = torch::ones_like(noiseX) * 5;
    auto X = baseX + noiseX;
    lv.loadPriorDist(5.0,1.0,1.0,1.0);
    lv.setTrainMode();
    auto xSize=X.sizes()[1];
    int epochs = 10;
    int batch_size = 128;
    for (int i=0;i<epochs;i++)
    {
         
         for(int k=0;k<num_samples-batch_size;k+=batch_size)
         {
          auto x = X.narrow(/*dim=*/0, /*start=*/k, /*length=*/batch_size);
          lv.learnStep(x);
         }
        
         cout<<"epoch="<<i<<"loss="<<lv.resultLoss/xSize<<endl;
    }
   
    //lv.getDimension();
    return 0;
}