/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model.h"
#include "loss.h"
#include "utils.h"

#include <assert.h>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace fasttext {

    Model::State::State(int32_t hiddenSize, int32_t outputSize, int32_t seed)
            : lossValue_(0.0),
              lossValueHyper_(0.0),
              nexamples_(0),
              nexamplesTree_(0),
              hidden(hiddenSize),
              output(outputSize),
              grad(hiddenSize),
              gradHyper(hiddenSize + 1),
              outVectarget(hiddenSize),
              outVeccenter(hiddenSize),
              inVectarget(hiddenSize),
              rng(seed),
              TotalSum(0),
              CurrentKappa(1.0),
              omega(0),
              DicId(0),
              IfSecondOrder(false),
              SampleCount(0){}

    real Model::State::getLoss() const {
        return lossValue_ / nexamples_;
    }

    real Model::State::getLossHyper() const {
        return lossValueHyper_ / nexamplesTree_;
    }

    real Model::State::getLossRegular() const {
        return lossValueRegular_;
    }

    void Model::State::incrementNExamples(real loss) {
        lossValue_ += loss;
        nexamples_++;
    }

    void Model::State::incrementNExamplesRegular(real loss) {
        lossValueRegular_ = lossValueRegular_*0.9 + loss*0.1;
    }

    void Model::State::incrementNExamplesHyper(real loss) {
        lossValueHyper_ += loss;
        nexamplesTree_++;
    }

    Model::Model(
            std::shared_ptr<Matrix> wi,
            std::shared_ptr<Matrix> wo,
            std::shared_ptr<Loss> loss,
            bool normalizeGradient)
            : wi_(wi), wo_(wo), loss_(loss), normalizeGradient_(normalizeGradient) {}


    void Model::predict(
            const std::vector<int32_t>& input,
            int32_t k,
            real threshold,
            Predictions& heap,
            State& state) const {
        if (k == Model::kUnlimitedPredictions) {
            k = wo_->size(0); // output size
        } else if (k <= 0) {
            throw std::invalid_argument("k needs to be 1 or higher!");
        }
        heap.reserve(k + 1);
        computeHidden(input, state);

        loss_->predict(k, threshold, heap, state);
    }

    void Model::computeHidden(const std::vector<int32_t>& input, State& state)
    const {
        Vector& hidden = state.hidden;
        hidden.zero();
        for (auto it = input.cbegin(); it != input.cend(); ++it) {
            hidden.addRow(*wi_, *it);
            state.inVectarget.addRow(*wi_,*it);
            state.DicId = *it;
        }
        hidden.mul(1.0 / input.size());
    }

    void Model::update(
            const std::vector<int32_t>& InputDicId,
            const std::vector<int32_t>& line,
            int32_t targetIndex,
            real lr,
            State& state) {
        if (InputDicId.size() == 0) {
            return;
        }
        computeHidden(InputDicId, state);
        //state.CurrentKappa = state.kappa[*input.cbegin()];
        //real NormHidden = state.hidden.norm();
        //计算当前in向量被抽中的概率
        //real kappa = 1000;
        //real Ck = pow(kappa,(100/2-1)) / pow(2*3.1416,100/2) / exp(logbesseli(100/2-1, kappa));
        //real ProbV = 0;
        //real ProbVhat = 0;
        //for (int32_t i=0; i < wo_->size(0); i++) {
            //ProbVhat = Ck * exp(kappa*wi_->CosSim(state.hidden, i));
            //ProbV += state.freq[i]*ProbVhat;
        //}ty
        Vector& grad = state.grad;
        grad.zero();
        real lossValue = loss_->forward(line, targetIndex, state, lr, true);
        state.incrementNExamples(lossValue);
        if (normalizeGradient_) {
            grad.mul(1.0 / InputDicId.size());
        }
        for (auto it = InputDicId.cbegin(); it != InputDicId.cend(); ++it) {
            wi_->addVectorToRow(grad, *it, 1.0);
        }
    }

    void Model::updateHyper(
            int32_t inWordId,
            int32_t outWordId,
            real lr,
            State& state) {
        // 取出input vector
        state.hidden.addRow(*wi_,inWordId);
        // 进行负采样并更新context vector与input vector
        real lossValue = loss_->forwardHyper(*wi_, inWordId ,outWordId, state, lr, true);
        state.incrementNExamplesHyper(lossValue);
        wi_->expMapToRow(state.gradHyper, inWordId);
    }

    void Model::updateRegular(
            int minibatch,
            real hyperparam,
            const std::vector<int32_t>& input,
            const std::vector<int32_t>& targets,
            int32_t targetIndex,
            real lr,
            State& state) {
        // 取出当前两个样本的Output vector的序号
//    std::cerr << "\rI am here! " << std::endl;
        std::vector<int32_t > SumOutVecIds;
        for (auto it = input.cbegin(); it != input.cend(); ++it) {
            SumOutVecIds.push_back(*it);
        }
        int32_t target = targets[targetIndex];
        SumOutVecIds.push_back(target);
        real batchloss = 0;
        // 进行in向量采样并更新对应的input vector
        for (int i=0; i < minibatch; i++){
            real lossValue = loss_->forwardRegular(minibatch, hyperparam, SumOutVecIds, wo_, wi_, lr, state,true);
            batchloss += lossValue;
        }
        state.incrementNExamplesRegular(batchloss);
//    state.TotalSum *= 0.9;
    }

    real Model::std_log(real x) const {
        return std::log(x + 1e-5);
    }

} // namespace fasttext
