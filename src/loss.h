/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <random>
#include <vector>

#include "matrix.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"
#include "dictionary.h"

namespace fasttext {

    class Loss {
    private:
        void findKBest(
                int32_t k,
                real threshold,
                Predictions& heap,
                const Vector& output) const;

    protected:
        std::vector<real> t_sigmoid_;
        std::vector<real> t_log_;
        std::shared_ptr<Matrix>& wo_;
        std::vector<entry> &words_;
        int64_t ntokens_;

        real log(real x) const;
        real sigmoid(real x) const;
    public:
        explicit Loss(std::shared_ptr<Matrix>& wo, std::vector<entry> & words, int64_t ntokens);
        virtual ~Loss() = default;
        real Beta(int m);
        real ReparameterizeOmega(int m, real kappa);
        void ReparameterizeZ(int m, Vector & Z, Vector & VcTemp, real omega);
        real ReparameterizeVc(int64_t m, real kappa, Vector & Z, Vector & Vc);

        virtual real forward(
                const std::vector<int32_t>& targets,
                int32_t targetIndex,
                Model::State& state,
                real lr,
                bool backprop) = 0;
        virtual real forwardHyper(
                fasttext::Matrix &wi_,
                int32_t inWordId,
                int32_t targetId,
                fasttext::Model::State &state,
                fasttext::real lr,
                bool backprop) = 0;
        virtual real forwardRegular(
                int minibatch,
                real hyperparam,
                std::vector<int32_t>& SumOutVecIds,
                std::shared_ptr<fasttext::Matrix> &wo,
                std::shared_ptr<fasttext::Matrix> &wi,
                fasttext::real lr,
                Model::State& state,
                bool backprop) = 0;
        virtual void computeOutput(Model::State& state) const = 0;
        virtual void predict(
                int32_t /*k*/,
                real /*threshold*/,
                Predictions& /*heap*/,
                Model::State& /*state*/) const;
        double logbesseli(double nu, double x) {
            double frac = x/nu;
            double square = 1 + frac*frac;
            double root = sqrt( square );
            double eta = root + log(frac) - log(1+root);
            double approx = -1 * log( sqrt( 2 * 3.1416* nu) ) + nu*eta - 0.25*log(square);
            return approx;
        }
    };

    class BinaryLogisticLoss : public Loss {
    protected:
        virtual real binaryLogistic(
                int32_t target,
                Model::State& state,
                bool labelIsPositive,
                real lr,
                bool backprop) const;

    public:
        explicit BinaryLogisticLoss(std::shared_ptr<Matrix>& wo, std::vector<entry> & words, int64_t ntokens);
        virtual ~BinaryLogisticLoss() noexcept override = default;
        void computeOutput(Model::State& state) const override;
    };

    class NegativeSamplingLoss : public BinaryLogisticLoss {
    protected:
        static const int32_t NEGATIVE_TABLE_SIZE = 10000000;

        int neg_;
        std::vector<int32_t> negatives_;
        std::uniform_int_distribution<size_t> uniform_;
        virtual int32_t getNegative(int32_t target, std::minstd_rand& rng);

    public:
        explicit NegativeSamplingLoss(
                std::shared_ptr<Matrix>& wo,
                std::vector<entry> & words,
                int64_t ntokens,
                int neg,
                const std::vector<int64_t>& targetCounts);
        ~NegativeSamplingLoss() noexcept override = default;

        real forward(
                const std::vector<int32_t>& targets,
                int32_t targetIndex,
                Model::State& state,
                real lr,
                bool backprop) override ;
        real forwardHyper(
                fasttext::Matrix &wi_,
                int32_t inWordId,
                int32_t targetId,
                fasttext::Model::State &state,
                fasttext::real lr,
                bool backprop) override {};
        real forwardRegular(
                int minibatch,
                real hyperparam,
                std::vector<int32_t>& SumOutVecIds,
                std::shared_ptr<fasttext::Matrix> &wo,
                std::shared_ptr<fasttext::Matrix> &wi,
                fasttext::real lr,
                Model::State& state,
                bool backprop) override {};
    };

    class InUnitLoss: public  NegativeSamplingLoss {
    protected:
        virtual real binaryLogistic (
                int32_t  target,
                Model::State& state,
                real uNorm,
                bool labelIsPositive,
                real lr,
                bool backprop ) const ;
        
    public:
        explicit InUnitLoss(
                std::shared_ptr<Matrix>& wo,
                std::vector<entry> & words,
                int64_t ntokens,
                int neg,
                const std::vector<int64_t>& targetCounts);
        ~InUnitLoss() noexcept override = default;
        real forward(
                const std::vector<int32_t> &targets,
                int32_t targetIndex,
                Model::State &state,
                real lr,
                bool backprop) override ;
    };
//        virtual real forward(
//                const std::vector<int32_t>& targets,
//                int32_t targetIndex,
//                Model::State& state,
//                real lr,
//                bool backprop) override;
//        real forwardHyper(
//                Matrix & wi_,
//                int32_t inWordId,
//                int32_t outWordId,
//                Model::State& state,
//                real lr,
//                bool backprop) override;
//        real forwardHyper(
//                Matrix & wi_,
//                int32_t inputId,
//                int32_t outWordId,
//                Model::State& state,
//                real lr,
//                bool backprop) override;
//        virtual real forwardHyper(
//                Matrix & wi_,
//                int32_t inWordId,
//                int32_t outWordId,
//                Model::State& state,
//                real lr,
//                bool backprop) = 0;
//class UnitBiLogisticLoss: public Loss {
//protected:
//    real unitBiLogistic(
//            int32_t  target,
//            Model::State& state,
//            real uNorm,
//            bool labelIsPositive,
//            real lr,
//            bool backprop) const;
//
//public:
//    explicit UnitBiLogisticLoss(std::shared_ptr<Matrix>& wo);
//    virtual ~UnitBiLogisticLoss() noexcept override = default;
//    void computeOutput(Model::State& state) const override;
//};
    class TreeInUnitLoss: public InUnitLoss {
    protected:
        int negTree_ = neg_ - 3;
        int32_t getNegativeHyper(
                int32_t inputId,
                int32_t target,
                std::minstd_rand& rng);
    public:
        explicit TreeInUnitLoss(
                std::shared_ptr<Matrix>& wo,
                std::vector<entry> & words,
                int64_t ntokens,
                int neg,
                const std::vector<int64_t>& targetCounts);
        ~TreeInUnitLoss() noexcept override = default;

        real forwardHyper (
                Matrix & wi_,
                int32_t inWordId,
                int32_t targetId,
                Model::State& state,
                real lr,
                bool backprop) override ;
    };

    class InUnitRegularLoss: public InUnitLoss {
    protected:
    public:
        explicit InUnitRegularLoss(
                std::shared_ptr<Matrix>& wo,
                std::vector<entry> & words,
                int64_t ntokens,
                int neg,
                const std::vector<int64_t>& targetCounts);
        ~InUnitRegularLoss() noexcept override = default;

        real forwardRegular (
                int minibatch,
                real hyperparam,
                std::vector<int32_t>& SumOutVecIds,
                std::shared_ptr<fasttext::Matrix> &wo,
                std::shared_ptr<fasttext::Matrix> &wi,
                real lr,
                Model::State& state,
                bool backprop) override ;
    };

    class OneVsAllLoss : public BinaryLogisticLoss {
    public:
        explicit OneVsAllLoss(std::shared_ptr<Matrix>& wo,
                              std::vector<entry> & words,
                              int64_t ntokens);
        ~OneVsAllLoss() noexcept override = default;
        real forward(
                const std::vector<int32_t>& targets,
                int32_t targetIndex,
                Model::State& state,
                real lr,
                bool backprop) override;
    };

    class HierarchicalSoftmaxLoss : public BinaryLogisticLoss {
    protected:
        struct Node {
            int32_t parent;
            int32_t left;
            int32_t right;
            int64_t count;
            bool binary;
        };

        std::vector<std::vector<int32_t>> paths_;
        std::vector<std::vector<bool>> codes_;
        std::vector<Node> tree_;
        int32_t osz_;
        void buildTree(const std::vector<int64_t>& counts);
        void dfs(
                int32_t k,
                real threshold,
                int32_t node,
                real score,
                Predictions& heap,
                const Vector& hidden) const;

    public:
        explicit HierarchicalSoftmaxLoss(
                std::shared_ptr<Matrix>& wo,
                std::vector<entry> & words,
                int64_t ntokens,
                const std::vector<int64_t>& counts);
        ~HierarchicalSoftmaxLoss() noexcept override = default;
        real forward(
                const std::vector<int32_t>& targets,
                int32_t targetIndex,
                Model::State& state,
                real lr,
                bool backprop) override;
        void predict(
                int32_t k,
                real threshold,
                Predictions& heap,
                Model::State& state) const override;
    };

    class SoftmaxLoss : public Loss {
    public:
        explicit SoftmaxLoss(std::shared_ptr<Matrix>& wo,
                             std::vector<entry> & words,
                             int64_t ntokens);
        ~SoftmaxLoss() noexcept override = default;
        real forward(
                const std::vector<int32_t>& targets,
                int32_t targetIndex,
                Model::State& state,
                real lr,
                bool backprop) override;
        void computeOutput(Model::State& state) const override;
    };

} // namespace fasttext
