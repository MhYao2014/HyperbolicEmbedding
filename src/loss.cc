/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "loss.h"
#include "utils.h"
#include <math.h>
#include <cmath>
#include <iostream>

namespace fasttext {

    constexpr int64_t SIGMOID_TABLE_SIZE = 512;
    constexpr int64_t MAX_SIGMOID = 8;
    constexpr int64_t LOG_TABLE_SIZE = 512;

    bool comparePairs(
            const std::pair<real, int32_t>& l,
            const std::pair<real, int32_t>& r) {
        return l.first > r.first;
    }

    real std_log(real x) {
        return std::log(x + 1e-5);
    }

    Loss::Loss(std::shared_ptr<Matrix>& wo) : wo_(wo) {
        t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
        for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
            real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
            t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
        }

        t_log_.reserve(LOG_TABLE_SIZE + 1);
        for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
            real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
            t_log_.push_back(std::log(x));
        }
    }

    real Loss::log(real x) const {
        if (x > 1.0) {
            return 0.0;
        }
        int64_t i = int64_t(x * LOG_TABLE_SIZE);
        return t_log_[i];
    }

    real Loss::sigmoid(real x) const {
        if (x < -MAX_SIGMOID) {
            return 0.0;
        } else if (x > MAX_SIGMOID) {
            return 1.0;
        } else {
            int64_t i =
                    int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
            return t_sigmoid_[i];
        }
    }

    void Loss::predict(
            int32_t k,
            real threshold,
            Predictions& heap,
            Model::State& state) const {
        computeOutput(state);
        findKBest(k, threshold, heap, state.output);
        std::sort_heap(heap.begin(), heap.end(), comparePairs);
    }

    void Loss::findKBest(
            int32_t k,
            real threshold,
            Predictions& heap,
            const Vector& output) const {
        for (int32_t i = 0; i < output.size(); i++) {
            if (output[i] < threshold) {
                continue;
            }
            if (heap.size() == k && std_log(output[i]) < heap.front().first) {
                continue;
            }
            heap.push_back(std::make_pair(std_log(output[i]), i));
            std::push_heap(heap.begin(), heap.end(), comparePairs);
            if (heap.size() > k) {
                std::pop_heap(heap.begin(), heap.end(), comparePairs);
                heap.pop_back();
            }
        }
    }

    BinaryLogisticLoss::BinaryLogisticLoss(std::shared_ptr<Matrix>& wo)
            : Loss(wo) {}

//    real UnitBiLogisticLoss::unitBiLogistic(
//            int32_t target,
//            fasttext::Model::State &state,
//            real uNorm,
//            bool labelIsPositive,
//            fasttext::real lr,
//            bool backprop) const {
//        real innerProduct = wo_->dotRow(state.hidden, target);
//        real score = sigmoid( (real)(innerProduct / uNorm) );
//        if (backprop) {
//            real alpha = lr * (real(labelIsPositive) - score);
//            // update v
//            wo_->addVectorToRow(state.hidden, target, (real)(alpha / uNorm));
//            // calculate the first term of u's grad
//            state.grad.addRow(*wo_, target, (real)(alpha / uNorm));
//            // calculate the second term of u's grad
//            state.grad.addVector(state.hidden, (real)(- alpha * innerProduct / (real)pow(uNorm, 3)));
//        }
//        if (labelIsPositive){
//            return -log(score);
//        } else {
//            return -log(1.0 - score);
//        }
//    }

    real BinaryLogisticLoss::binaryLogistic(
            int32_t target,
            Model::State& state,
            bool labelIsPositive,
            real lr,
            bool backprop) const {
        real score = sigmoid(wo_->dotRow(state.hidden, target));
        if (backprop) {
            real alpha = lr * (real(labelIsPositive) - score);
            state.grad.addRow(*wo_, target, alpha);
            wo_->addVectorToRow(state.hidden, target, alpha);
        }
        if (labelIsPositive) {
            return -log(score);
        } else {
            return -log(1.0 - score);
        }
    }

//    void UnitBiLogisticLoss::computeOutput(fasttext::Model::State &state) const {
//        Vector& output = state.output;
//        real uNorm = state.hidden.norm();
//        output.mul(*wo_, state.hidden);
//        output.mul((real)(1.0 / uNorm));
//        int32_t osz = output.size();
//        for(int32_t i = 0; i < osz; i++) {
//            output[i] = sigmoid(output[i]);
//        }
//    }

    void BinaryLogisticLoss::computeOutput(Model::State& state) const {
        Vector& output = state.output;
        output.mul(*wo_, state.hidden);
        int32_t osz = output.size();
        for (int32_t i = 0; i < osz; i++) {
            output[i] = sigmoid(output[i]);
        }
    }

//    int32_t UnitNegativeSamplingLoss::getNegative(
//            int32_t target,
//            std::minstd_rand &rng) {
//        int32_t negative;
//        do {
//            negative = negatives_[uniform_(rng)];
//        } while (target == negative);
//        return negative;
//    }

    int32_t NegativeSamplingLoss::getNegative(
            int32_t target,
            std::minstd_rand& rng) {
        int32_t negative;
        do {
            negative = negatives_[uniform_(rng)];
        } while (target == negative);
        return negative;
    }

    NegativeSamplingLoss::NegativeSamplingLoss(
            std::shared_ptr<Matrix>& wo,
            int neg,
            const std::vector<int64_t>& targetCounts)
            : BinaryLogisticLoss(wo), neg_(neg), negatives_(), uniform_() {
        real z = 0.0;
        for (size_t i = 0; i < targetCounts.size(); i++) {
            z += pow(targetCounts[i], 0.5);
        }
        for (size_t i = 0; i < targetCounts.size(); i++) {
            real c = pow(targetCounts[i], 0.5);
            for (size_t j = 0; j < c * NegativeSamplingLoss::NEGATIVE_TABLE_SIZE / z;
                 j++) {
                negatives_.push_back(i);
            }
        }
        uniform_ = std::uniform_int_distribution<size_t>(0, negatives_.size() - 1);
    }

//    real UnitNegativeSamplingLoss::forward(
//            const std::vector<int32_t> &targets,
//            int32_t targetIndex,
//            Model::State &state,
//            real lr,
//            bool backprop) {
//        assert( targetIndex >= 0 );
//        assert( targetIndex < targets.size() );
//        int32_t target = targets[targetIndex];
//        real uNorm = state.hidden.norm();
//        real loss = unitBiLogistic(target, state, uNorm, true, lr, backprop);
//
//        for(int32_t i = 0; i < neg_; i++) {
//            auto negativeTarget = getNegative(target, state.rng);
//            loss += unitBiLogistic(negativeTarget, state, uNorm, false, lr, backprop);
//        }
//        return loss;
//    }
//
//    real UnitNegativeSamplingLoss::forwardHyper(
//            Matrix & wi_,
//            int32_t inWordId,
//            int32_t targetId,
//            Model::State& state,
//            real lr,
//            bool backprop) {
//        assert( targetId >= 0 );
//        real loss = 0;
//        Vector target(wi_.size(1));
//        target.zero();
//        target.addRow(wi_, targetId);
//        // 生成u^{\hat}与v_{+}^{\hat}
//        Vector uHat(state.hidden.size()+1);
//        uHat.generateFrom(state.hidden);
//        Vector targetHat(target.size()+1);
//        targetHat.generateFrom(target);
//        // 计算正样本的洛仑兹内积;
//        real lorentzProduct = uHat.lorentzPro(targetHat);
//        assert(-lorentzProduct > 1);
//        real numeraPos = exp(-acosh(-lorentzProduct));
//        // 计算hidden与正例的欧式梯度
//        state.gradHyper = targetHat;
//        Vector hHatTarget = uHat;
//        // 计算梯度前的系数并乘上度规矩阵的逆，从而得到双曲梯度
//        real diffCoff = 1 / std::sqrt(lorentzProduct*lorentzProduct - 1);
//        hHatTarget.mul(-diffCoff);
//        state.gradHyper.mul(-diffCoff);
//        // 对正例利用双曲梯度进行黎曼SGD
//        hHatTarget.proj(targetHat);
//        hHatTarget.mul(-lr);
//        wi_.expMapToRow(hHatTarget, targetId);
//        // 负采样并计算梯度，最后进行黎曼SGD
////        std::vector<real> loreProVec;
////        std::vector<int64_t> negIdVec;
////        // 第一遍遍历先将每个负样本遍历完，并记录下每个负样本与hidden的内积
////        for (int32_t i = 0; i < neg_; i++) {
////            auto negativeTarget = getNegativeHyper(inWordId, targetId, state.rng);
////            negIdVec.push_back(negativeTarget);
////            // 抽取负样本的原始词向量
////            target.zero();
////            target.addRow(wi_, negativeTarget);
////            // 生成负样本的双曲向量
////            targetHat.generateFrom(target);
////            // 计算负样本的洛伦茨内积
////            lorentzProduct = uHat.lorentzPro(targetHat);
////            loreProVec.push_back(lorentzProduct);
////        }
////        // 第二遍遍历负样本并计算分子分母
////        real denomi = 0;
////        real numerTemp = 0;
////        std::vector<real> numer;
////        for (int32_t i = 0; i < neg_; i++) {
////            assert(-loreProVec[i] > 1);
////            // 计算分子
////            numerTemp = exp(-acosh(-loreProVec[i]));
////            // 梯度计算中出现的负号留在了这里相乘
////            numer.push_back(-numerTemp);
////            // 累加分母
////            denomi += numerTemp;
////        }
//        // 计算损失
//        loss += -log(numeraPos);
//        // 第三遍遍历负样本并更新梯度
////        Vector hHatHiddenTemp(targetHat.size());
////        for (int32_t i = 0; i < neg_; i++) {
////            // 计算负样本的欧式梯度因子
////            diffCoff = numer[i] / denomi / std::sqrt((loreProVec[i]*loreProVec[i]) - 1);
////            // 负样本的欧式梯度是从uHat开始的
////            hHatTarget.zero();
////            hHatTarget = uHat;
////            hHatTarget.mul(-diffCoff);
////            // 取出负样本的原始词向量
////            target.zero();
////            target.addRow(wi_, negIdVec[i]);
////            targetHat.generateFrom(target);
////            // 开始做黎曼SGD
////            hHatTarget.proj(targetHat);
////            hHatTarget.mul(-lr);
////            wi_.expMapToRow(hHatTarget, negIdVec[i]);
////            // 计算hidden的欧式梯度并累加
////            hHatHiddenTemp.zero();
////            hHatHiddenTemp = targetHat;
////            hHatHiddenTemp.mul(-diffCoff);
////            state.gradHyper.addVector(hHatHiddenTemp);
////        }
//        state.gradHyper.proj(uHat);
//        state.gradHyper.mul(-lr);
//        return loss;
//    }

    real NegativeSamplingLoss::forward(
            const std::vector<int32_t>& targets,
            int32_t targetIndex,
            Model::State& state,
            real lr,
            bool backprop) {
        assert(targetIndex >= 0);
        assert(targetIndex < targets.size());
        int32_t target = targets[targetIndex];
        real loss = binaryLogistic(target, state, true, lr, backprop);

        for (int32_t n = 0; n < neg_; n++) {
            auto negativeTarget = getNegative(target, state.rng);
            loss += binaryLogistic(negativeTarget, state, false, lr, backprop);
        }
        return loss;
    }

    InUnitLoss::InUnitLoss(
            std::shared_ptr<fasttext::Matrix> &wo,
            int neg,
            const std::vector<int64_t> &targetCounts)
            : NegativeSamplingLoss(wo, neg, targetCounts){}

    real InUnitLoss::binaryLogistic (
            int32_t  target,
            Model::State& state,
            real uNorm,
            bool labelIsPositive,
            real lr,
            bool backprop ) const {
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

    real InUnitLoss::forward(
            const std::vector<int32_t> &targets,
            int32_t targetIndex,
            Model::State &state,
            real lr,
            bool backprop) {
//        std::cerr << "\rI am here ! The forward" << std::endl;
        assert( targetIndex >= 0 );
        assert( targetIndex < targets.size() );
        int32_t target = targets[targetIndex];
        real uNorm = state.hidden.norm();
        real loss = InUnitLoss::binaryLogistic(target, state, uNorm, true, lr, backprop);

        for(int32_t i = 0; i < neg_; i++) {
            auto negativeTarget = getNegative(target, state.rng);
            loss += InUnitLoss::binaryLogistic(negativeTarget, state, uNorm, false, lr, backprop);
        }
        return loss;
    }

    InUnitRegularLoss::InUnitRegularLoss(
            std::shared_ptr<fasttext::Matrix> &wo,
            int neg,
            const std::vector<int64_t> &targetCounts)
            : InUnitLoss(wo, neg, targetCounts){}

    void InUnitRegularLoss::forwardRegular(
            std::vector<int32_t>& SumOutVecIds,
            std::shared_ptr<fasttext::Matrix> &wo,
            std::shared_ptr<fasttext::Matrix> &wi,
            fasttext::real lr,
            Model::State& state,
            bool backprop) {
        Vector SumOutVec(wo->size(1));
        SumOutVec.zero();
        for (auto it = SumOutVecIds.cbegin(); it != SumOutVecIds.cend(); ++it) {
            SumOutVec.addRow(*wo, *it);
        }
        std::minstd_rand rng(0);
        int32_t RegularInVecId = negatives_[uniform_(state.rng)];
        Vector RegularInVec(wi->size(1));
        RegularInVec.zero();
        RegularInVec.addRow(*wi, RegularInVecId);
        real RegularInVecNorm = RegularInVec.norm();
        real InnerProduct = SumOutVec.dotmul(RegularInVec);
        RegularInVec.elemul(RegularInVec);
        RegularInVec.elemul(SumOutVec);
        RegularInVec.mul(pow(1/RegularInVecNorm,3));
        SumOutVec.mul(1/RegularInVecNorm);
        SumOutVec.substract(RegularInVec);
        wi->addVectorToRow(SumOutVec, RegularInVecId, -1*lr*std::exp(InnerProduct)*(1/1000000));
    }

    TreeInUnitLoss::TreeInUnitLoss(
            std::shared_ptr<fasttext::Matrix> &wo,
            int neg,
            const std::vector<int64_t> &targetCounts)
            : InUnitLoss(wo, neg, targetCounts){}

    int32_t TreeInUnitLoss::getNegativeHyper(
            int32_t inputId,
            int32_t target,
            std::minstd_rand &rng) {
        int32_t negative;
        do {
            negative = negatives_[uniform_(rng)];
            if (negative == inputId) {
                continue;
            }
        } while (target == negative || inputId == negative);
        return negative;
    }

    real TreeInUnitLoss::forwardHyper(
            fasttext::Matrix &wi_,
            int32_t inWordId,
            int32_t targetId,
            fasttext::Model::State &state,
            fasttext::real lr,
            bool backprop) {
        assert( targetId >= 0 );
        real loss = 0;
        Vector target(wi_.size(1));
        target.zero();
        target.addRow(wi_, targetId);
        // 生成u^{\hat}与v_{+}^{\hat}
        Vector uHat(state.hidden.size()+1);
        uHat.generateFrom(state.hidden);
        Vector targetHat(target.size()+1);
        targetHat.generateFrom(target);
        // 计算正样本的洛仑兹内积;
        real lorentzProduct = uHat.lorentzPro(targetHat);
        assert(-lorentzProduct > 1);
        real numeraPos = exp(-acosh(-lorentzProduct));
        // 计算hidden与正例的欧式梯度
        state.gradHyper = targetHat;
        Vector hHatTarget = uHat;
        // 计算梯度前的系数并乘上度规矩阵的逆，从而得到双曲梯度
        real diffCoff = 1 / std::sqrt(lorentzProduct*lorentzProduct - 1);
        hHatTarget.mul(-diffCoff);
        state.gradHyper.mul(-diffCoff);
        // 对正例利用双曲梯度进行黎曼SGD
        hHatTarget.proj(targetHat);
        hHatTarget.mul(-lr);
        wi_.expMapToRow(hHatTarget, targetId);
        // 负采样并计算梯度，最后进行黎曼SGD
        std::vector<real> loreProVec;
        std::vector<int64_t> negIdVec;
        // 第一遍遍历先将每个负样本遍历完，并记录下每个负样本与hidden的内积
        for (int32_t i = 0; i < neg_; i++) {
            auto negativeTarget = TreeInUnitLoss::getNegativeHyper(inWordId, targetId, state.rng);
            negIdVec.push_back(negativeTarget);
            // 抽取负样本的原始词向量
            target.zero();
            target.addRow(wi_, negativeTarget);
            // 生成负样本的双曲向量
            targetHat.generateFrom(target);
            // 计算负样本的洛伦茨内积
            lorentzProduct = uHat.lorentzPro(targetHat);
            loreProVec.push_back(lorentzProduct);
        }
        // 第二遍遍历负样本并计算分子分母
        real denomi = 0;
        real numerTemp = 0;
        std::vector<real> numer;
        for (int32_t i = 0; i < neg_; i++) {
            assert(-loreProVec[i] > 1);
            // 计算分子
            numerTemp = exp(-acosh(-loreProVec[i]));
            // 梯度计算中出现的负号留在了这里相乘
            numer.push_back(-numerTemp);
            // 累加分母
            denomi += numerTemp;
        }
        // 计算损失
        loss += -log(numeraPos);
//         第三遍遍历负样本并更新梯度
        Vector hHatHiddenTemp(targetHat.size());
        for (int32_t i = 0; i < neg_; i++) {
            // 计算负样本的欧式梯度因子
            diffCoff = numer[i] / denomi / std::sqrt((loreProVec[i]*loreProVec[i]) - 1);
            // 负样本的欧式梯度是从uHat开始的
            hHatTarget.zero();
            hHatTarget = uHat;
            hHatTarget.mul(-diffCoff);
            // 取出负样本的原始词向量
            target.zero();
            target.addRow(wi_, negIdVec[i]);
            targetHat.generateFrom(target);
            // 开始做黎曼SGD
            hHatTarget.proj(targetHat);
            hHatTarget.mul(-lr);
            wi_.expMapToRow(hHatTarget, negIdVec[i]);
            // 计算hidden的欧式梯度并累加
            hHatHiddenTemp.zero();
            hHatHiddenTemp = targetHat;
            hHatHiddenTemp.mul(-diffCoff);
            state.gradHyper.addVector(hHatHiddenTemp);
        }
        state.gradHyper.proj(uHat);
        state.gradHyper.mul(-lr);
        return loss;
    }

    void HierarchicalSoftmaxLoss::buildTree(const std::vector<int64_t>& counts) {
        tree_.resize(2 * osz_ - 1);
        for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
            tree_[i].parent = -1;
            tree_[i].left = -1;
            tree_[i].right = -1;
            tree_[i].count = 1e15;
            tree_[i].binary = false;
        }
        for (int32_t i = 0; i < osz_; i++) {
            tree_[i].count = counts[i];
        }
        int32_t leaf = osz_ - 1;
        int32_t node = osz_;
        for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
            int32_t mini[2] = {0};
            for (int32_t j = 0; j < 2; j++) {
                if (leaf >= 0 && tree_[leaf].count < tree_[node].count) {
                    mini[j] = leaf--;
                } else {
                    mini[j] = node++;
                }
            }
            tree_[i].left = mini[0];
            tree_[i].right = mini[1];
            tree_[i].count = tree_[mini[0]].count + tree_[mini[1]].count;
            tree_[mini[0]].parent = i;
            tree_[mini[1]].parent = i;
            tree_[mini[1]].binary = true;
        }
        for (int32_t i = 0; i < osz_; i++) {
            std::vector<int32_t> path;
            std::vector<bool> code;
            int32_t j = i;
            while (tree_[j].parent != -1) {
                path.push_back(tree_[j].parent - osz_);
                code.push_back(tree_[j].binary);
                j = tree_[j].parent;
            }
            paths_.push_back(path);
            codes_.push_back(code);
        }
    }

    real HierarchicalSoftmaxLoss::forward(
            const std::vector<int32_t>& targets,
            int32_t targetIndex,
            Model::State& state,
            real lr,
            bool backprop) {
        real loss = 0.0;
        int32_t target = targets[targetIndex];
        const std::vector<bool>& binaryCode = codes_[target];
        const std::vector<int32_t>& pathToRoot = paths_[target];
        for (int32_t i = 0; i < pathToRoot.size(); i++) {
            loss += binaryLogistic(pathToRoot[i], state, binaryCode[i], lr, backprop);
        }
        return loss;
    }

    void HierarchicalSoftmaxLoss::predict(
            int32_t k,
            real threshold,
            Predictions& heap,
            Model::State& state) const {
        dfs(k, threshold, 2 * osz_ - 2, 0.0, heap, state.hidden);
        std::sort_heap(heap.begin(), heap.end(), comparePairs);
    }

    void HierarchicalSoftmaxLoss::dfs(
            int32_t k,
            real threshold,
            int32_t node,
            real score,
            Predictions& heap,
            const Vector& hidden) const {
        if (score < std_log(threshold)) {
            return;
        }
        if (heap.size() == k && score < heap.front().first) {
            return;
        }

        if (tree_[node].left == -1 && tree_[node].right == -1) {
            heap.push_back(std::make_pair(score, node));
            std::push_heap(heap.begin(), heap.end(), comparePairs);
            if (heap.size() > k) {
                std::pop_heap(heap.begin(), heap.end(), comparePairs);
                heap.pop_back();
            }
            return;
        }

        real f = wo_->dotRow(hidden, node - osz_);
        f = 1. / (1 + std::exp(-f));

        dfs(k, threshold, tree_[node].left, score + std_log(1.0 - f), heap, hidden);
        dfs(k, threshold, tree_[node].right, score + std_log(f), heap, hidden);
    }

    SoftmaxLoss::SoftmaxLoss(std::shared_ptr<Matrix>& wo) : Loss(wo) {}

    void SoftmaxLoss::computeOutput(Model::State& state) const {
        Vector& output = state.output;
        output.mul(*wo_, state.hidden);
        real max = output[0], z = 0.0;
        int32_t osz = output.size();
        for (int32_t i = 0; i < osz; i++) {
            max = std::max(output[i], max);
        }
        for (int32_t i = 0; i < osz; i++) {
            output[i] = exp(output[i] - max);
            z += output[i];
        }
        for (int32_t i = 0; i < osz; i++) {
            output[i] /= z;
        }
    }

    real SoftmaxLoss::forward(
            const std::vector<int32_t>& targets,
            int32_t targetIndex,
            Model::State& state,
            real lr,
            bool backprop) {
        computeOutput(state);

        assert(targetIndex >= 0);
        assert(targetIndex < targets.size());
        int32_t target = targets[targetIndex];

        if (backprop) {
            int32_t osz = wo_->size(0);
            for (int32_t i = 0; i < osz; i++) {
                real label = (i == target) ? 1.0 : 0.0;
                real alpha = lr * (label - state.output[i]);
                state.grad.addRow(*wo_, i, alpha);
                wo_->addVectorToRow(state.hidden, i, alpha);
            }
        }
        return -log(state.output[target]);
    };

    OneVsAllLoss::OneVsAllLoss(std::shared_ptr<Matrix>& wo)
            : BinaryLogisticLoss(wo) {}

    real OneVsAllLoss::forward(
            const std::vector<int32_t>& targets,
            int32_t /* we take all targets here */,
            Model::State& state,
            real lr,
            bool backprop) {
        real loss = 0.0;
        int32_t osz = state.output.size();
        for (int32_t i = 0; i < osz; i++) {
            bool isMatch = utils::contains(targets, i);
            loss += binaryLogistic(i, state, isMatch, lr, backprop);
        }

        return loss;
    }

} // namespace fasttext

