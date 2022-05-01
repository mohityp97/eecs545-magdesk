#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
using namespace std;
using namespace Eigen;

int main()
{
    Vector3d v(0, 1, 2);
    float theta = 0 * M_PI, phy = 0 * M_PI, beta = -1 * M_PI;
    Quaterniond q(cos(0.5 * beta), sin(0.5 * beta) * sin(theta) * cos(phy), sin(0.5 * beta) * sin(theta) * sin(phy), sin(0.5 * beta) * cos(theta));
    cout << "Co-effs: " << q.coeffs() << endl;
    cout << "Norm: " << q.norm() << endl;
    q.normalize();
    cout << "Co-effs: " << q.coeffs() << endl;
    cout << "Norm: " << q.norm() << endl;

    cout << "We can now use it to rotate a vector" << std::endl
         << v << " to " << endl
         << q.inverse() * v << endl; //convert world coord to local coord
    return 0;
}