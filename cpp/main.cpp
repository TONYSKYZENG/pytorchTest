#include <LinearVAE.hpp>
#include <vector>
using namespace std;
using namespace TROCHPACK_VAE;
int main()
{
    LinearVAE lv;
    lv.loadModule("linearVAE.pt");
    std::vector<float> vc = {5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.4,5.2};
    lv.runForward(vc);
    std::cout<<"mu="<<lv.resultMu<<",sigma="<<lv.resultSigma<<endl;
    int num_samples = 10;
    int input_dim = 10;
    auto noiseX = torch::randn({num_samples, input_dim});
    auto baseX = torch::ones_like(noiseX) * 5;
    auto X = baseX + noiseX;
    lv.loadPriorDist(5.0,1.0,1.0,1.0);
    lv.setTrainMode();
    auto xSize=X.sizes()[0]*X.sizes()[1];
    for (int i=0;i<5;i++)
    {
         lv.learnStep(X);
         cout<<"i="<<i<<"loss="<<lv.resultLoss/xSize<<endl;
    }
   
    //lv.getDimension();
    return 0;
}