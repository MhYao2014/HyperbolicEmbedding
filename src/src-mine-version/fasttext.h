//
// Created by hutao on 19-4-5.
//
#pragma once

#include <time.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <queue>
#include <set>
#include <tuple>

#include "args.h"
#include "densematrix.h"
#include "matrix.h"
#include "vector.h"
#include "utils.h"
#include "model.h"
#include "dictionary.h"
#include "real.h"
#ifndef FASTTEXTMINE_FASTTEXT_H
#define FASTTEXTMINE_FASTTEXT_H

#endif //FASTTEXTMINE_FASTTEXT_H
namespace fasttext {
    class FastText {
    protected:
        std::shared_ptr<Args> args_;
        std::shared_ptr<Dictionary> dict_;
        std::shared_ptr<Matrix> input_;
        std::shared_ptr<Matrix> output_;
        std::shared_ptr<Model> model_;
        std::atomic<int64_t> tokenCount_{};
        std::atomic<real> loss_{};
        std::chrono::steady_clock::time_point start_;
        bool quant_;
        std::unique_ptr<DenseMatrix> wordVectors_;
        std::shared_ptr<Matrix> createRandomMatrix() const;
        std::shared_ptr<Matrix> createTrainOutputMatrix() const;
        std::shared_ptr<Loss> createLoss(std::shared_ptr<Matrix>& output, std::shared_ptr<Matrix>& input);
        std::vector<int64_t> getTargetCounts() const;
        void startThreads();
        void trainThread(int32_t);
        void printInfo(real, real, std::ostream&);
        void skipgram(Model::State& state, real lr, const std::vector<int32_t>& line);
        void signModel(std::ostream&);
        void addInputVector(Vector&, int32_t) const;
        void addOutputVector(Vector&, int32_t) const;
    public:
        FastText();
        const Args getArgs() const;
        void train(const Args& args);
        void getWordVector(Vector& vec, const std::string& word) const;
        void getOutputVector(Vector& vec, const std::string& word) const;
        void saveModel(const std::string& filename);
        void saveVectors(const std::string& filename);
        void saveOutput(const std::string& filename);
    };
}