//
// Created by hutao on 19-4-2.
//
#pragma once

#include <memory>
#include <random>
#include <vector>
#include "matrix.h"
#include "model.h"
#include "vector.h"
#include "utils.h"
#include "real.h"
#ifndef FASTTEXTMINE_LOSS_H
#define FASTTEXTMINE_LOSS_H

#endif //FASTTEXTMINE_LOSS_H

namespace fasttext{
    class Model;

    class Loss{
    private:
//        void findBest(
//                int32_t k,
//                real threshold,
//                Predictions &heap,
//                const Vector& output) const;

    protected:
        std::vector<real> t_sigmoid_;
        std::vector<real> t_log_;
        std::shared_ptr<Matrix>& wo_;
        std::shared_ptr<Matrix>& wi_;

        real log(real x) const;
        real sigmoid(real x) const;

    public:
        explicit Loss(std::shared_ptr<Matrix>& wo, std::shared_ptr<Matrix>& wi);
        virtual ~Loss() = default;

        virtual real forward(
                const std::vector<int32_t>& targets,
                int32_t targetIndex,
                Model::State &state,
                real lr,
                bool backprop) = 0;

        virtual void computOutput(Model::State & state) const = 0;

//        virtual void predict(
//                int32_t /*k*/,
//                real /*threshold*/,
//                Predictions& /*heap*/,
//                Model::State& /*state*/) const;
    };

    class UnitBiLogisticLoss: public Loss {
    protected:
        real unitBiLogistic(
                int32_t  target,
                Model::State& state,
                real uNorm,
                bool labelIsPositive,
                real lr,
                bool backprop) const;

    public:
        explicit UnitBiLogisticLoss(std::shared_ptr<Matrix>& wo, std::shared_ptr<Matrix>& wi);
        virtual ~UnitBiLogisticLoss() noexcept override = default;
        void computOutput(Model::State& state) const override;
    };

    class BinaryLogisticLoss: public Loss{
    protected:
        real binaryLogistic(
                int32_t target,
                Model::State& state,
                bool labelIsPositive,
                real lr,
                bool backprop) const;

    public:
        explicit BinaryLogisticLoss(std::shared_ptr<Matrix>& wo, std::shared_ptr<Matrix>& wi);
        virtual ~BinaryLogisticLoss() noexcept override = default;
        void computOutput(Model::State& state) const override;
    };

    class UnitNegativeSamplingLoss: public  UnitBiLogisticLoss {
    protected:
        static const int32_t UNITNEGATIVE_TABLE_SIZE = 10000000;

        int neg_;
        std::vector<int32_t> negatives_;
        std::uniform_int_distribution<size_t> uniform_;
        int32_t getNegative(int32_t target, std::minstd_rand& rng);
    public:
        explicit UnitNegativeSamplingLoss(
                std::shared_ptr<Matrix>& wo,
                std::shared_ptr<Matrix>& wi,
        int neg,
        const std::vector<int64_t>& targetCounts);
        ~UnitNegativeSamplingLoss() noexcept override = default;

        real forward(
                const std::vector<int32_t>& targets,
                int32_t targetIndex,
                Model::State& state,
                real lr,
                bool backprop) override;
    };

    class NegativeSamplingLoss: public BinaryLogisticLoss {
    protected:
        static const int32_t NEGATIVE_TABLE_SIZE = 10000000;

        int neg_;
        std::vector<int32_t> negatives_;
        std::uniform_int_distribution<size_t> uniform_;
        int32_t getNegative(int32_t target, std::minstd_rand& rng);

    public:
        explicit NegativeSamplingLoss(
                std::shared_ptr<Matrix>& wo,
                std::shared_ptr<Matrix>& wi,
                int neg,
                const std::vector<int64_t>& targetCounts);
        ~NegativeSamplingLoss() noexcept override = default;

        real forward(
                const std::vector<int32_t>& targets,
                int32_t targetIndex,
                Model::State& state,
                real lr,
                bool backprop) override;
    };


}
