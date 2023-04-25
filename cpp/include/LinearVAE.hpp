/*! \file LinearVAE.h*/
//Copyright (C) 2022 by the IntelliStream team (https://github.com/intellistream)
// Created by tony on 03/03/23.
//
#ifndef _VAE_LINEARVAE_HPP_
#define _VAE_LINEARVAE_HPP_
#pragma once
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <memory>
namespace TROCHPACK_VAE {
#define  newtorchMethod std::make_shared<torch::jit::Method>
typedef std::shared_ptr<torch::jit::Method> torchMethodPtr;
#define  newtorchOptimiser std::make_shared<torch::optim::Adam>
typedef std::shared_ptr<torch::optim::Adam> torchOptimiserPtr;
/**
* @defgroup TROCHPACK_VAE The VAE warp classes
* @{
 * This package covers the warp function to call libtorch VAE
*/
/**
 * @ingroup TROCHPACK_VAE
 * @class LinearVAE  LinearVAE.hpp
 * @brief The class to call a LinearVAE
*/
class LinearVAE
{
private:
    /* data */
    torch::jit::script::Module module;
    uint64_t inputDimension,latentDimension;
    torchMethodPtr getDimensionMethod,getMuEstimationMethod,loadPriorDistMethod;
    torchMethodPtr lossUnderNormalMethod;
    void getDimension();
    void getMuEstimation();
    torchOptimiserPtr myOpt;
    //
public:
    LinearVAE(){

    }
    ~LinearVAE(){

    }
   void setEvalMode(){
    module.eval();
   }
   void setTrainMode(){
    module.train();
   }
    /**
     *  @brief load the module from [path]
     * @param path the string to indicate the loaded path
    */
    void loadModule(std::string path);
    /**
     * @brief run an NN forward on data
     * @param data the data stored in plain std vector, will be automatically converted to the suitable shape
     * @note please read the @ref resultMu and @ref resultSigma after this
    */
    void runForward(std::vector<float> data);
    float resultMu,resultSigma;
      /**
     * @brief load the prior distribution
     * @param pmu
     * @param psigma
     * @param a0
     * @param b0
    */
    void loadPriorDist(float pmu,float psigma,float a0,float b0);
     /**
     * @brief to perform one step learning on data
     * @param data the data stored in plain std vector, will be automatically converted to the suitable shape
    */
    void learnStep(std::vector<float> data);
    /**
     * @brief to perform one step learning on data
     * @param data the data stored as tensor, should check size outside
    */
    void learnStep(torch::Tensor data);
    float resultLoss;
};

/**
 *@}
*/

}

#endif