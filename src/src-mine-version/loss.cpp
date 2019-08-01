//
// Created by hutao on 19-4-3.
//
#include "loss.h"
#include "utils.h"
#include <cmath>

namespace fasttext {
    constexpr int64_t SIGMOID_TABLE_SIZE = 512;
    constexpr int64_t MAX_SIGMOID = 8;
    constexpr int64_t LOG_TABLE_SIZE = 512;

    real Loss::log(fasttext::real x) const {
        if (x > 1.0){
            return 0.0;
        }
        int64_t  i = int64_t(x*LOG_TABLE_SIZE);
        return t_log_[i];
    }

    real Loss::sigmoid(fasttext::real x) const {
        if (x < - MAX_SIGMOID){
            return 0.0;
        } else if (x > MAX_SIGMOID){
            return 1.0;
        } else {
            int64_t i = int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
            return t_sigmoid_[i];
        }
    }

    Loss::Loss(std::shared_ptr<fasttext::Matrix> &wo, std::shared_ptr<Matrix>& wi) : wo_(wo), wi_(wi) {
        t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
        for (int i = 0; i < SIGMOID_TABLE_SIZE; i++){
            real x = real((i*2*MAX_SIGMOID)) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
            t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
        }
        t_log_.reserve(LOG_TABLE_SIZE + 1);
        for (int i = 0; i < LOG_TABLE_SIZE; i++){
            real x = (real(i)+1e-5) / LOG_TABLE_SIZE;
            t_log_.push_back(std::log(x));
        }
    }

    UnitBiLogisticLoss::UnitBiLogisticLoss(std::shared_ptr<Matrix> &wo, std::shared_ptr<Matrix>& wi) : Loss(wo, wi) {}

    real UnitBiLogisticLoss::unitBiLogistic(
            int32_t target,
            fasttext::Model::State &state,
            real uNorm,
            bool labelIsPositive,
            fasttext::real lr,
            bool backprop) const {
        real innerProduct = wo_->dotRow(state.hidden, target);
        real score = sigmoid( (real)(innerProduct / uNorm) );
        if (backprop) {
            real alpha = lr * (real(labelIsPositive) - score);
            // update v
            wo_->addVectorToRow(state.hidden, target, (real)(alpha / uNorm));
            // calculate the first term of u's grad
            state.grad.addRow(*wo_, target, (real)(alpha / uNorm));
            // calculate the second term of u's grad
            state.grad.addVector(state.hidden, (real)(- alpha * innerProduct / (real)pow(uNorm, 3)));
        }
        if (labelIsPositive){
            return -log(score);
        } else {
            return -log(1.0 - score);
        }
    }

    void UnitBiLogisticLoss::computOutput(fasttext::Model::State &state) const {
        Vector& output = state.output;
        real uNorm = state.hidden.norm();
        output.mul(*wo_, state.hidden);
        output.mul((real)(1.0 / uNorm));
        int32_t osz = output.size();
        for(int32_t i = 0; i < osz; i++) {
            output[i] = sigmoid(output[i]);
        }
    }

    BinaryLogisticLoss::BinaryLogisticLoss(std::shared_ptr<fasttext::Matrix> &wo, std::shared_ptr<Matrix>& wi)
    : Loss(wo, wi) {}

    real BinaryLogisticLoss::binaryLogistic(
            int32_t target,
            Model::State &state,
            bool labelIsPositive,
            real lr,
            bool backprop) const {
        real score = sigmoid(wo_->dotRow(state.hidden, target));
        real scoreOut = sigmoid(wi_->dotRow(state.hiddenOut, target));
        if (backprop) {
            real alpha = lr * (real(labelIsPositive) - score);
            real alphaOut = lr * (real(labelIsPositive) - scoreOut);
            state.grad.addRow(*wo_, target, alpha);
            state.gradOut.addRow(*wi_, target, alphaOut);
            wo_->addVectorToRow(state.hidden, target, alpha);
            wi_->addVectorToRow(state.hiddenOut, target, alphaOut);
        }
        if (labelIsPositive){
            return -log(std::min(score, scoreOut));
        } else {
            return -log(1.0 - std::max(score, scoreOut));
        }
    }

    void BinaryLogisticLoss::computOutput(fasttext::Model::State &state) const {
        Vector& output = state.output;
        output.mul(*wo_, state.hidden);
        int32_t osz = output.size();
        for(int32_t i = 0; i < osz; i++) {
            output[i] = sigmoid(output[i]);
        }
    }

    int32_t NegativeSamplingLoss::getNegative(
            int32_t target,
            std::minstd_rand &rng) {
        int32_t negative;
        do {
            negative = negatives_[uniform_(rng)];
        } while (target == negative);
        return negative;
    }

    int32_t UnitNegativeSamplingLoss::getNegative(
            int32_t target,
            std::minstd_rand &rng) {
        int32_t negative;
        do {
            negative = negatives_[uniform_(rng)];
        } while (target == negative);
        return negative;
    }

    NegativeSamplingLoss::NegativeSamplingLoss(
            std::shared_ptr<fasttext::Matrix> &wo,
            std::shared_ptr<fasttext::Matrix> & wi,
            int neg,
            const std::vector<int64_t> &targetCounts)
            : BinaryLogisticLoss(wo, wi), neg_(neg), negatives_(), uniform_(){
        real z = 0.0;
        for(size_t i = 0; i < targetCounts.size(); i++) {
            z += pow(targetCounts[i], 0.5);
        }
        for(size_t i = 0; i < targetCounts.size(); i++){
            real c = pow(targetCounts[i], 0.5);
            for (int32_t j = 0; j < c*NegativeSamplingLoss::NEGATIVE_TABLE_SIZE / z; j++) {
                negatives_.push_back(i);
            }
        }
        uniform_ = std::uniform_int_distribution<size_t> (0, negatives_.size() - 1);
    }

    UnitNegativeSamplingLoss::UnitNegativeSamplingLoss(
            std::shared_ptr<fasttext::Matrix> &wo,
            std::shared_ptr<fasttext::Matrix> & wi,
            int neg,
            const std::vector<int64_t> &targetCounts)
            : UnitBiLogisticLoss(wo, wi), neg_(neg), negatives_(), uniform_(){
        real z = 0.0;
        for(size_t i = 0; i < targetCounts.size(); i++) {
            z += pow(targetCounts[i], 0.5);
        }
        for(size_t i = 0; i < targetCounts.size(); i++){
            real c = pow(targetCounts[i], 0.5);
            for (int32_t j = 0; j < c*UnitNegativeSamplingLoss::UNITNEGATIVE_TABLE_SIZE / z; j++) {
                negatives_.push_back(i);
            }
        }
        uniform_ = std::uniform_int_distribution<size_t> (0, negatives_.size() - 1);
    }

    real NegativeSamplingLoss::forward(
            const std::vector<int32_t> &targets,
            int32_t targetIndex,
            Model::State &state,
            real lr,
            bool backprop) {
        assert( targetIndex >= 0 );
        assert( targetIndex < targets.size() );
        int32_t target = targets[targetIndex];
        real loss = binaryLogistic(target, state, true, lr, backprop);

        for(int32_t i = 0; i < neg_; i++) {
            auto negativeTarget = getNegative(target, state.rng);
            loss += binaryLogistic(negativeTarget, state, false, lr, backprop);
        }
        return loss;
    }

    real UnitNegativeSamplingLoss::forward(
            const std::vector<int32_t> &targets,
            int32_t targetIndex,
            Model::State &state,
            real lr,
            bool backprop) {
        assert( targetIndex >= 0 );
        assert( targetIndex < targets.size() );
        int32_t target = targets[targetIndex];
        real uNorm = state.hidden.norm();
        real loss = unitBiLogistic(target, state, uNorm, true, lr, backprop);

        for(int32_t i = 0; i < neg_; i++) {
            auto negativeTarget = getNegative(target, state.rng);
            loss += unitBiLogistic(negativeTarget, state, uNorm, false, lr, backprop);
        }
        return loss;
    }


}