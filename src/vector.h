/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <ostream>
#include <vector>

#include "real.h"

namespace fasttext {

class Matrix;

class Vector {
 protected:
  std::vector<real> data_;
  real lorentzProduct(Vector& x);

 public:
  explicit Vector(int64_t);
  Vector(const Vector&) = default;
  Vector(Vector&&) noexcept = default;
  Vector& operator=(const Vector&) = default;
  Vector& operator=(Vector&&) = default;

  inline real* data() {
    return data_.data();
  }
  inline const real* data() const {
    return data_.data();
  }
  inline real& operator[](int64_t i) {
    return data_[i];
  }
  inline const real& operator[](int64_t i) const {
    return data_[i];
  }

  inline int64_t size() const {
    return data_.size();
  }
  void zero();
  void generateFrom(Vector &);
  void hyperInverse();
  void mul(real);
  real lorentzPro(Vector& x);
  real norm() const;
  void proj(Vector& x);
  void addVector(const Vector& source);
  void addVector(const Vector&, real);
  void addRow(const Matrix&, int64_t);
  void addRow(const Matrix&, int64_t, real);
  void mul(const Matrix&, const Vector&);
  void elemul(const Vector&);
  real dotmul(const Vector&);
  void substract(const Vector&);
  int64_t argmax();
};

std::ostream& operator<<(std::ostream&, const Vector&);

} // namespace fasttext
