#include <iostream>

#include "robust_pca.h"

using Eigen::MatrixXd;

int main() {
  MatrixXd D = MatrixXd::Random(5, 5);
  D = D.array() * D.array();

  MatrixXd A = MatrixXd::Zero(5, 5);
  MatrixXd E = MatrixXd::Zero(5, 5);

  std::cout << "The original matirix; D = \n" << D << std::endl;

  // Perform Robust PCA
  sp::ml::robust_pca(D, A, E);
  std::cout << "Estimated row rank matrix: A = \n" << A << std::endl;
  std::cout << "Estimated sparse matrix: E = \n" << E << std::endl;

  std::cout << "Reconstructed matrix: A + E =:\n" << A + E << std::endl;

  std::cout << "Reconstruction Error = " << (D - (A + E)).norm() << std::endl;

  return 0;
}
