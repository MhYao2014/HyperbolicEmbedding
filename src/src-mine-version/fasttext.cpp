//
// Created by hutao on 19-4-7.
//
#include "fasttext.h"
#include "loss.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fasttext {

    constexpr int32_t FASTTEXT_VERSION = 12; /* Version 1b */
    constexpr int32_t FASTTEXT_FILEFORMAT_MAGIC_INT32 = 793712314;

    FastText::FastText() :quant_(false), wordVectors_(nullptr) {}

    const Args FastText::getArgs() const {
        return *args_.get();
    }

    std::vector<int64_t> FastText::getTargetCounts() const {
        if (args_->model == model_name::sup) {
            return dict_->getCounts(entry_type::label);
        } else {
            return dict_->getCounts(entry_type::word);
        }
    }

    std::shared_ptr<Matrix> FastText::createRandomMatrix() const {
        std::shared_ptr<DenseMatrix> input = std::make_shared<DenseMatrix>(
                dict_->nwords() + args_->bucket, args_->dim);
        input->uniform(1.0 / args_->dim);

        return input;
    }

    std::shared_ptr<Matrix> FastText::createTrainOutputMatrix() const {
        int64_t m =
                (args_->model == model_name::sup) ? dict_->nlabels() : dict_->nwords();
        std::shared_ptr<DenseMatrix> output =
                std::make_shared<DenseMatrix>(m, args_->dim);
        output->zero();

        return output;
    }

    std::shared_ptr<Loss> FastText::createLoss(std::shared_ptr<Matrix>& output, std::shared_ptr<Matrix>& input) {
        loss_name lossName = args_->loss;
        switch (lossName) {
            case loss_name::ns:
                return std::make_shared<NegativeSamplingLoss>(
                        output, input, args_->neg, getTargetCounts());
            case loss_name::uns:
                return std::make_shared<UnitNegativeSamplingLoss>(
                        output, input, args_->neg, getTargetCounts());
        }
    }

    void FastText::train(const Args &args) {
        args_ = std::make_shared<Args>(args);
        dict_ = std::make_shared<Dictionary>(args_);
        if (args_->input == "-") {
            throw std::invalid_argument("Cannot use stdin for training!");
        }
        std::ifstream ifs(args_->input);
        if (!ifs.is_open()) {
            throw std::invalid_argument(
                    args_->input + " cannot be opened for training!");
        }
        dict_->readFromFile(ifs);
        ifs.close();
        input_ = createRandomMatrix();
        output_ = createTrainOutputMatrix();
        auto loss = createLoss(output_, input_);
        bool normalizeGradient = (args_->model == model_name::sup);
        model_ = std::make_shared<Model>(input_, output_, loss, normalizeGradient);
        startThreads();
    }

    void FastText::startThreads() {
        start_ = std::chrono::steady_clock::now();
        tokenCount_ = 0;
        loss_ = -1;
        std::vector<std::thread> threads;
        for (int32_t i = 0; i < args_->thread; i++) {
            threads.push_back(std::thread([=]() { trainThread(i); }));
        }
        const int64_t ntokens = dict_->ntokens();
        // Same condition as trainThread
        while (tokenCount_ < args_->epoch * ntokens) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (loss_ >= 0 && args_->verbose > 1) {
                real progress = real(tokenCount_) / (args_->epoch * ntokens);
                std::cerr << "\r";
                printInfo(progress, loss_, std::cerr);
            }
        }
        for (int32_t i = 0; i < args_->thread; i++) {
            threads[i].join();
        }
        if (args_->verbose > 0) {
            std::cerr << "\r";
            printInfo(1.0, loss_, std::cerr);
            std::cerr << std::endl;
        }
    }

    void FastText::trainThread(int32_t threadId) {
        std::ifstream ifs(args_->input);
        utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);
        Model::State state(args_->dim, output_->size(0), threadId);
        const int64_t ntokens = dict_->ntokens();
        int64_t localTokenCount = 0;
        std::vector<int32_t> line, labels;
        while (tokenCount_ < args_->epoch * ntokens) {
            real progress = real(tokenCount_) / (args_->epoch * ntokens);
            real lr = args_->lr * (1.0 - progress);
            localTokenCount += dict_->getLine(ifs, line, state.rng);
            skipgram(state, lr, line);
            if (localTokenCount > args_->lrUpdateRate) {
                tokenCount_ += localTokenCount;
                localTokenCount = 0;
                if (threadId == 0 && args_->verbose > 1)
                    loss_ = state.getLoss();
            }
        }
        if (threadId == 0) {
            loss_ = state.getLoss();
        }
        ifs.close();
    }

    void FastText::skipgram(
            Model::State& state,
            real lr,
            const std::vector<int32_t>& line) {
        std::uniform_int_distribution<> uniform(1, args_->ws);
        for (int32_t w = 0; w < line.size(); w++) {
            int32_t boundary = uniform(state.rng);
            const std::vector<int32_t>& ngrams = dict_->getSubwords(line[w]);
            for (int32_t c = -boundary; c <= boundary; c++) {
                if (c != 0 && w + c >= 0 && w + c < line.size()) {
                    model_->update(ngrams, line, w + c, lr, state);
                }
            }
        }
    }

    void FastText::printInfo(real progress, real loss, std::ostream& log_stream) {
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        double t =
                std::chrono::duration_cast<std::chrono::duration<double>>(end - start_)
                        .count();
        double lr = args_->lr * (1.0 - progress);
        double wst = 0;

        int64_t eta = 2592000; // Default to one month in seconds (720 * 3600)

        if (progress > 0 && t >= 0) {
            progress = progress * 100;
            eta = t * (100 - progress) / progress;
            wst = double(tokenCount_) / t / args_->thread;
        }
        int32_t etah = eta / 3600;
        int32_t etam = (eta % 3600) / 60;

        log_stream << std::fixed;
        log_stream << "Progress: ";
        log_stream << std::setprecision(1) << std::setw(5) << progress << "%";
        log_stream << " words/sec/thread: " << std::setw(7) << int64_t(wst);
        log_stream << " lr: " << std::setw(9) << std::setprecision(6) << lr;
        log_stream << " loss: " << std::setw(9) << std::setprecision(6) << loss;
        log_stream << " ETA: " << std::setw(3) << etah;
        log_stream << "h" << std::setw(2) << etam << "m";
        log_stream << std::flush;
    }

    void FastText::signModel(std::ostream& out) {
        const int32_t magic = FASTTEXT_FILEFORMAT_MAGIC_INT32;
        const int32_t version = FASTTEXT_VERSION;
        out.write((char*)&(magic), sizeof(int32_t));
        out.write((char*)&(version), sizeof(int32_t));
    }

    void FastText::saveModel(const std::string& filename) {
        std::ofstream ofs(filename, std::ofstream::binary);
        if (!ofs.is_open()) {
            throw std::invalid_argument(filename + " cannot be opened for saving!");
        }
        signModel(ofs);
        args_->save(ofs);
        dict_->save(ofs);

        ofs.write((char*)&(quant_), sizeof(bool));
        input_->save(ofs);

        ofs.write((char*)&(args_->qout), sizeof(bool));
        output_->save(ofs);

        ofs.close();
    }

    void FastText::addInputVector(Vector& vec, int32_t ind) const {
        vec.addRow(*input_, ind);
    }

    void FastText::addOutputVector(Vector& vec, int32_t ind) const {
        vec.addRow(*output_, ind);
    }

    void FastText::getWordVector(Vector& vec, const std::string& word) const {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
        vec.zero();
        for (int i = 0; i < ngrams.size(); i++) {
            addInputVector(vec, ngrams[i]);
        }
        if (ngrams.size() > 0) {
            vec.mul(1.0 / ngrams.size());
        }
    }

    void FastText::getOutputVector(fasttext::Vector &vec, const std::string &word) const {
        const std::vector<int32_t>& ngrams = dict_->getSubwords(word);
        vec.zero();
        for (int i = 0; i < ngrams.size(); i++) {
            addOutputVector(vec, ngrams[i]);
        }
        if (ngrams.size() > 0) {
            vec.mul(1.0 / ngrams.size());
        }
    }

    void FastText::saveOutput(const std::string& filename) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            throw std::invalid_argument(
                    filename + " cannot be opened for saving vectors!");
        }
        if (quant_) {
            throw std::invalid_argument(
                    "Option -saveOutput is not supported for quantized models.");
        }
        int32_t n =
                (args_->model == model_name::sup) ? dict_->nlabels() : dict_->nwords();
        ofs << n << " " << args_->dim << std::endl;
        Vector vec(args_->dim);
        for (int32_t i = 0; i < n; i++) {
            std::string word = (args_->model == model_name::sup) ? dict_->getLabel(i)
                                                                 : dict_->getWord(i);
            vec.zero();
            vec.addRow(*output_, i);
            ofs << word << " " << vec << std::endl;
        }
        ofs.close();
    }

    void FastText::saveVectors(const std::string& filename) {
        std::ofstream ofsInput(filename + "Input");
//        std::ofstream ofsOutput(filename + "Output");
        if (!ofsInput.is_open()) {
            throw std::invalid_argument(
                    filename + " cannot be opened for saving vectors!");
        }
//        if (!ofsOutput.is_open()) {
//            throw std::invalid_argument(
//                    filename + " cannot be opened for saving vectors!");
//        }
        ofsInput << dict_->nwords() << " " << args_->dim << std::endl;
//        ofsOutput << dict_->nwords() << " " << args_->dim << std::endl;
        Vector vec(args_->dim);
//        Vector vecOutput(args_->dim);
        for (int32_t i = 0; i < dict_->nwords(); i++) {
            std::string word = dict_->getWord(i);
            getWordVector(vec, word);
//            getOutputVector(vecOutput, word);
//            std::cout << std::endl;
//            std::cout << "i am here !" << std::endl;
            ofsInput << word << " " << vec << std::endl;
//            ofsOutput << word << " " << vecOutput << std::endl;

        }
        ofsInput.close();
//        ofsOutput.close();
    }
}