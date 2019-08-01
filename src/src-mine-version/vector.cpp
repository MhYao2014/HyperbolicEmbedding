//
// Created by hutao on 19-4-2.
//
#include "vector.h"

#include <assert.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>
#include "matrix.h"

namespace fasttext {
    Vector::Vector(int64_t m): data_(m) {};

    void Vector::zero() {
        std::fill(data_.begin(), data_.end(), 0.0);
    }
    void Vector::mul(real a) {
        for(int64_t i = 0; i < size(); i++){
            data_[i] *= a;
        }
    }
    real Vector::norm() const {
        real sum = 0;
        for(int64_t i = 0; i < size(); i++){
            sum += data_[i] * data_[i];
        }
        return std::sqrt(sum);
    }
    void Vector::addVector(const fasttext::Vector &source) {
        assert(size() == source.size());
        for (int64_t i = 0; i < size(); i++){
            data_[i] += source.data_[i];
        }
    }
    void Vector::addVector(const fasttext::Vector &source, fasttext::real s) {
        assert(size() == source.size());
        for (int64_t i = 0; i < size(); i++){
            data_[i] += s*source.data_[i];
        }
    }
    void Vector::addRow(const fasttext::Matrix & A, int64_t i) {
        assert( i >= 0);
        assert( i < A.size(0));
        assert(size() == A.size(1));
        A.addRowToVector(*this, i);
    }
    void Vector::addRow(const fasttext::Matrix & A, int64_t i, fasttext::real a) {
        assert( i >= 0 );
        assert( i < A.size(0));
        assert(size() == A.size(1));
        A.addRowToVector(*this, i, a);
    }
    void Vector::mul(const fasttext::Matrix & A, const fasttext::Vector & vec) {
        assert(A.size(0) == size());
        assert(A.size(1) == vec.size());
        for (int64_t i = 0; i < size(); i++){
            data_[i] = A.dotRow(vec, i);
        }
    }
    int64_t Vector::argmax() {
        real max = data_[0];
        int64_t argmax = 0;
        for (int64_t j = 1; j < size(); j++){
            if (data_[j] > max){
                max = data_[j];
                argmax = j;
            }
        }
        return argmax;
    }
    std::ostream&operator<<(std::ostream& os, const Vector& v){
        os << std::setprecision(5);
        for (int64_t j = 0; j < v.size(); j++){
            os << v[j] << ' ';
        }
        return os;
    }
}
