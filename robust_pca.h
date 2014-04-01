//  Copyright (c) 2014 Ryuichi Yamamoto
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to
//  deal in the Software without restriction, including without limitation the
//  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//  sell copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//  IN THE SOFTWARE.

#pragma once

#include <Eigen/Core>
#include <Eigen/SVD>
#include <cmath>
#include <iostream>

namespace sp {
namespace ml {

namespace internal {

template <class ValueType, class Vector>
inline int count_larger_than(const Vector& v, ValueType value) {
  int count = 0;

  for (int i = 0; i < v.size(); ++i) {
    if (v[i] > value) ++count;
  }

  return count;
}

/**
 * Robust Principal Component Analysis (RPCA) using the inexact augumented
 * Lagrance multiplier.
 * @param D observation matrix (D = A + E)
 * @param A  row-rank matrix
 * @param E  sparse matrix
 */
template <class ValueType = float>
void robust_pca_inexact_alm(
    Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>& D,
    Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>& A,
    Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>& E) {
  typedef typename Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>
      Matrix;
  typedef typename Eigen::Array<ValueType, Eigen::Dynamic, Eigen::Dynamic>
      Array;
  typedef typename Eigen::Matrix<ValueType, Eigen::Dynamic, 1> Vector;

  const int M = D.rows();
  const int N = D.cols();

  using std::cout;
  using std::endl;

  // supplementary variable
  Matrix Y = D;
  A = Matrix::Zero(M, N);
  E = Matrix::Zero(M, N);
  Array zero = Matrix::Zero(M, N);

  // parameters
  const double lambda = 1.0 / sqrt(std::max(M, N));
  const double rho = 1.5;

  Eigen::JacobiSVD<Matrix> svd_only_singlar_values(Y);
  const double norm_two =
      svd_only_singlar_values.singularValues()(0);  // can be tuned
  const double norm_inf = Y.array().abs().maxCoeff() / lambda;
  const double dual_norm = std::max(norm_two, norm_inf);
  const double d_norm = D.norm();
  Y /= dual_norm;

  double mu = 1.25 / norm_two;
  const double mu_bar = mu * 1.0e+7;

  bool converged = false;
  int max_iter = 1000;
  double error_tolerance = 1.0e-7;
  int iter = 0;
  int total_svd = 0;
  int sv = 10;
  while (!converged) {
    // update sparse matrix E
    Array temp_T = D - A + (1.0 / mu) * Y;
    E = (temp_T - lambda / mu).max(zero) + (temp_T + lambda / mu).min(zero);

    // force non-negative
    E = E.array().max(zero);  // 論文には書いてない
                              //cout << E << endl;

    // SVD
    Eigen::JacobiSVD<Matrix> svd(D - E + 1.0 / mu * Y,
                                 Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix U = svd.matrixU();
    Matrix V = svd.matrixV();
    Vector singularValues = svd.singularValues();

    // trancate dimention
    int svp = count_larger_than(singularValues, 1 / mu);
    if (svp < sv) {
      sv = std::min(svp + 1, N);
    } else {
      sv = std::min(svp + static_cast<int>(0.05 * N + 0.5), N);
    }

    // update A
    Matrix S_th =
        (singularValues.head(svp).array() - 1.0 / mu).matrix().asDiagonal();
    A = U.leftCols(svp) * S_th * V.leftCols(svp).transpose();

    // force non-negative
    A = A.array().max(zero);

    total_svd += 1;
    Matrix Z = D - A - E;
    Y = Y + mu * Z;
    mu = std::min(mu * rho, mu_bar);

    // objective function
    double objective = Z.norm() / d_norm;

    if (objective < error_tolerance) {
      converged = true;
    }

    if (++iter >= max_iter) {
      break;
    }
  }
}

}  // end namespace internal

/**
 * Interface of RPCA
 * @param D: observation matrix (D = A + E)
 * @param A: row-rank matrix
 * @param E: sparse matrix
*/
template <class ValueType = float>
void robust_pca(Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>& D,
                Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>& A,
                Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic>& E) {
  internal::robust_pca_inexact_alm(D, A, E);
}

}  // end namespace ml
}  // end namespace sp
