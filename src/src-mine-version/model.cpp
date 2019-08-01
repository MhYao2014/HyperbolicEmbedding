//
// Created by hutao on 19-4-3.
//
#include <assert.h>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include "model.h"
#include "loss.h"
#include "utils.h"

namespace fasttext{
    Model::Model(std::shared_ptr<fasttext::Matrix> wi,
                 std::shared_ptr<fasttext::Matrix> wo,
                 std::shared_ptr<fasttext::Loss> loss,
                 bool normalizeGradient)
                 :wi_(wi), wo_(wo), loss_(loss), normalizeGradient_(normalizeGradient) {}

    Model::State::State(int64_t hiddenSize, int64_t outputSize, int64_t seed)
    : lossValue_(0.0),
      nexamples_(0),
      hidden(hiddenSize),
      hiddenOut(hiddenSize),
      output(outputSize),
      outputOut(outputSize),
      grad(hiddenSize),
      gradOut(hiddenSize),
      rng(seed) {}

    real Model::State::getLoss() const {
        return lossValue_ / nexamples_;
    }

    void Model::State::incrementNExamples(fasttext::real loss) {
        lossValue_ += loss;
        nexamples_++;
    }

    void Model::computeHidden(
            const std::vector<int32_t> &input,
            fasttext::Model::State &state) {
        Vector &hidden = state.hidden;
        Vector &hiddenOut = state.hiddenOut;
        hidden.zero();
        hiddenOut.zero();
        for (auto it = input.cbegin(); it != input.cend(); ++it) {
            hidden.addRow(*wi_, *it);
            hiddenOut.addRow(*wo_, *it);
        }
        hidden.mul(1.0 / input.size());
    }

//    void Model::computeHiddenOut(
//            const std::vector<int32_t> &input,
//            fasttext::Model::State &state) {
//        Vector &hidden = state.hidden;
//        hidden.zero();
//        for (auto it = input.cbegin(); it != input.cend(); ++it) {
//            hidden.addRow(*wo_, *it);
//        }
//        hidden.mul(1.0 / input.size());
//    }

    void Model::update(
            const std::vector<int32_t>& input,
            const std::vector<int32_t>& targets,
            int32_t targetIndex,
            fasttext::real lr,
            fasttext::Model::State &state) {
        if (input.size() == 0){
            return;
        }
        // 以in向量作为输入建模
        computeHidden(input, state);

        Vector &grad = state.grad;
        Vector &gradOut = state.gradOut;
        grad.zero();
        gradOut.zero();
        real lossValue = loss_->forward(targets, targetIndex, state, lr, true);
        state.incrementNExamples(lossValue);

        if (normalizeGradient_) {
            grad.mul(1.0 / input.size());
        }
        for (auto it = input.cbegin(); it != input.cend(); ++it) {
            wi_->addVectorToRow(grad, *it, 1.0);
            wo_->addVectorToRow(gradOut, *it, 1.0);
        }
    }
}
