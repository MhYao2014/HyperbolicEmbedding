//
// Created by hutao on 19-4-3.
//
#pragma once

#include <memory>
#include <vector>
#include <random>
#include <utility>

#include "matrix.h"
#include "vector.h"
#include "utils.h"
#include "real.h"

namespace fasttext{
    class Loss;

    class Model {
    protected:
        std::shared_ptr<Matrix> wi_;
        std::shared_ptr<Matrix> wo_;
        std::shared_ptr<Loss> loss_;
        bool normalizeGradient_;

    public:
        Model(  std::shared_ptr<Matrix> wi,
                std::shared_ptr<Matrix> wo,
                std::shared_ptr<Loss> loss,
                bool normalizeGradient);
        Model(const Model& model) = delete;
        Model(Model&&model) = delete;
        Model&operator=(const Model& other) = delete;
        Model&operator=(Model&& other) = delete;

        class State {
        private:
            real lossValue_;
            int64_t nexamples_;

        public:
            Vector hidden;
            Vector hiddenOut;
            Vector output;
            Vector outputOut;
            Vector grad;
            Vector gradOut;
            std::minstd_rand rng;

            State(int64_t hiddenSize, int64_t outputSize, int64_t seed);
            real getLoss() const;
            void incrementNExamples(real loss);
        };

        void update(
                const std::vector<int32_t>& input,
                const std::vector<int32_t>& targets,
                int32_t targetIndex,
                real lr,
                State& state);

        void computeHidden(
                const std::vector<int32_t>& input,
                State& state);

        void computeHiddenOut(
                const std::vector<int32_t>& input,
                State& state);
    };
}