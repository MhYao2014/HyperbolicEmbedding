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
#include <utility>
#include <vector>

#include "matrix.h"
#include "real.h"
#include "utils.h"
#include "vector.h"
#include "dictionary.h"

namespace fasttext {

class Loss;

class Model {
 protected:
  std::shared_ptr<Matrix> wi_;
  std::shared_ptr<Matrix> wo_;
  std::shared_ptr<Loss> loss_;
  bool normalizeGradient_;

 public:
  Model(
      std::shared_ptr<Matrix> wi,
      std::shared_ptr<Matrix> wo,
      std::shared_ptr<Loss> loss,
      bool normalizeGradient);
  Model(const Model& model) = delete;
  Model(Model&& model) = delete;
  Model& operator=(const Model& other) = delete;
  Model& operator=(Model&& other) = delete;

  class State {
   private:
    real lossValue_;
    real lossValueHyper_;
    real lossValueRegular_;
    int64_t nexamples_;
    int64_t nexamplesTree_;

   public:
    Vector hidden;
    Vector output;
    Vector grad;
    Vector gradHyper;
      Vector VHat;
      Vector Z;
      Vector Vc;
    std::minstd_rand rng;
    std::vector<real> freq;
    std::vector<real> kappa;
    int64_t SampleCount;
    real TotalSum;
      real omega;
      real alpha;
//    int32_t input;

    State(int32_t hiddenSize, int32_t outputSize, int32_t seed);
    real getLoss() const;
    real getLossHyper() const;
    real getLossRegular() const;
    void incrementNExamples(real loss);
    void incrementNExamplesRegular(real loss);
    void incrementNExamplesHyper(real loss);
  };

  void predict(
      const std::vector<int32_t>& input,
      int32_t k,
      real threshold,
      Predictions& heap,
      State& state) const;
  void update(
      const std::vector<int32_t>& input,
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      real lr,
      State& state);
  void updateHyper(
          int32_t inWordId,
          int32_t outWordId,
          real lr,
          State& state);
  void updateRegular(
          int minibatch,
          real hyperparam,
          const std::vector<int32_t>& input,
          const std::vector<int32_t>& targets,
          int32_t targetIndex,
          real lr,
          State& state);
  void computeHidden(const std::vector<int32_t>& input, State& state) const;

  real std_log(real) const;

  static const int32_t kUnlimitedPredictions = -1;
  static const int32_t kAllLabelsAsTarget = -1;
    double logbesseli(double nu, double x) {
        double frac = x/nu;
        double square = 1 + frac*frac;
        double root = sqrt( square );
        double eta = root + log(frac) - log(1+root);
        double approx = -1 * log( sqrt( 2 * 3.1416* nu) ) + nu*eta - 0.25*log(square);
        return approx;
    }
};

} // namespace fasttext
