#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <math.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <thread>
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
const int kNumObservations = 8;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;

// -------------
// pure C++ code
// -------------
std::vector<double> multiply(const std::vector<double> &input)
{
    std::vector<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i)
        output[i] = 10 * static_cast<double>(input[i]);

    return output;
}

void multiply2(const std::vector<double> &input, std::vector<double> &output)
{
    // std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        output[i] = 10 * static_cast<double>(input[i]);
}

int add(int i, int j)
{
    return i + j;
}

struct Cost_FixedM_1mag
{
    Cost_FixedM_1mag(double x, double y, double z, double x2, double y2, double z2, double m) : Bx(x), By(y), Bz(z), Xs(x2), Ys(y2), Zs(z2), M{m} {} // init the sensor position and the magnitude reading.
    template <typename T>
    bool operator()(const T *const x, const T *const y, const T *const z, const T *const theta, const T *const phy, const T *const Gx, const T *const Gy, const T *const Gz, T *residual) const
    { // x y z is the coordinates of magnate j, m is the attributes of magate j, theta phy is the orientation of the magnate
        Eigen::Matrix<T, 3, 1> VecM = Eigen::Matrix<T, 3, 1>(sin(theta[0]) * cos(phy[0]), sin(theta[0]) * sin(phy[0]), cos(theta[0])) * 1e-7 * exp(M);
        Eigen::Matrix<T, 3, 1> VecR = Eigen::Matrix<T, 3, 1>(Xs - x[0], Ys - y[0], Zs - z[0]);
        T NormR = VecR.norm();
        Eigen::Matrix<T, 3, 1> B = (3.0 * VecR * (VecM.transpose() * VecR) / pow(NormR, 5) - VecM / pow(NormR, 3)); //convert it's unit to correspond with the input
        // std::cout << "B= " << (B(0, 0) + Gx[0]) * 1e6 << "\t" << (B(1, 0) + Gy[0]) * 1e6 << "\t" << (B(2, 0) + Gz[0]) * 1e6 << "\n";
        // std::cout << B(0) << '\n'
        //           << B(1) << '\n'
        //           << B(2) << std::endl;
        residual[0] = (B(0, 0) + Gx[0]) * 1e6 - Bx;
        residual[1] = (B(1, 0) + Gy[0]) * 1e6 - By;
        residual[2] = (B(2, 0) + Gz[0]) * 1e6 - Bz;
        // std::cout << residual[0] << '\t' << residual[1] << '\t' << residual[2] << std::endl;
        return true;
    }

private:
    const double Bx;
    const double By;
    const double Bz;
    const double Xs;
    const double Ys;
    const double Zs;
    const double M;
};

std::vector<double> solve_1mag(std::vector<double> readings, std::vector<double> pSensor, std::vector<double> init_param)
{
    // std::vector<float> test_vector = { 2,1,3 };
    // Eigen::MatrixXf readings_vec = Eigen::Map<Eigen::Matrix<double, 8, 3> >(readings.data());
    // Eigen::MatrixXf pSensor_vec = Eigen::Map<Eigen::Matrix<double, 8, 3> >(pSensor.data());
    Eigen::VectorXd readings_vec_1 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(readings.data(), readings.size());
    Eigen::VectorXd pSensor_vec_1 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(pSensor.data(), pSensor.size());
    // Eigen::MatrixXd readings_vec_1(&readings[0], 8, 3);
    // Eigen::MatrixXd pSensor_vec_1(&pSensor[0], 8, 3);
    Eigen::Map<Eigen::MatrixXd> readings_vec(readings_vec_1.data(), 3, 8);
    Eigen::Map<Eigen::MatrixXd> pSensor_vec(pSensor_vec_1.data(), 3, 8);
    readings_vec = readings_vec.transpose();
    pSensor_vec = pSensor_vec.transpose();
    std::cout
        << "readings_vec: " << readings_vec << "\n";
    std::cout << "pSensor_vec: " << pSensor_vec << "\n";

    // Eigen::Matrix<double, 8, 3> testdata;

    // testdata << 9., 22.2, -34.5, 8.7, 49.35, 34.65, 22.8, 43.5,
    //     6.6, 22.35, 26.7, -8.85, 27.9, 34.65, -5.25, 26.55,
    //     36.45, 6.15, 30.15, 38.25, 3.75, 29.55, 34.35, -2.7;

    // Eigen::Matrix<double, 8, 3> sPosition;
    // sPosition << 1, 1, 1,
    //     -1, 1, 1,
    //     -1, -1, 1,
    //     1, -1, 1,
    //     1, 1, -1,
    //     -1, 1, -1,
    //     -1, -1, -1,
    //     1, -1, -1;
    // sPosition *= 5e-2;

    // google::InitGoogleLogging(argv[0]);
    // double x = -0.04;
    // double y = 0.04;
    // double z = 0.04;
    double Gx = init_param[0];
    double Gy = init_param[1];
    double Gz = init_param[2];
    double m = init_param[3];
    double x = init_param[4];
    double y = init_param[5];
    double z = init_param[6];
    double theta = init_param[7];
    double phy = init_param[8];
    std::cout << "Initial x: " << x << " y: " << y << " z: " << z << " m: " << m << " theta: " << theta << " phy: " << phy << " Gx: " << Gx << " Gy: " << Gy << " Gz: " << Gz << "\n";
    Problem problem;
    for (int i = 0; i < pSensor_vec.rows(); ++i)
    {
        // problem.AddResidualBlock(
        //     new AutoDiffCostFunction<Cost, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
        //         new Cost(testdata(i, 0), testdata(i, 1), testdata(i, 2), sPosition(i, 0), sPosition(i, 1), sPosition(i, 2))),
        //     NULL, &x, &y, &z, &m, &theta, &phy, &Gx, &Gy, &Gz);

        problem.AddResidualBlock(
            new AutoDiffCostFunction<Cost_FixedM_1mag, 3, 1, 1, 1, 1, 1, 1, 1, 1>(
                new Cost_FixedM_1mag(readings_vec(i, 0), readings_vec(i, 1), readings_vec(i, 2), pSensor_vec(i, 0), pSensor_vec(i, 1), pSensor_vec(i, 2), m)),
            NULL, &x, &y, &z, &theta, &phy, &Gx, &Gy, &Gz);
    }
    Solver::Options options;
    // options.max_num_iterations = 1e6;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    // options.num_threads = std::thread::hardware_concurrency();
    options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    options.max_num_iterations = 1e5;
    // options.min_relative_decrease = 1e-16;
    // options.max_num_consecutive_invalid_steps = 1e6;
    // options.function_tolerance = 1e-32;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    // std::cout << "Initial x: " << 0.0 << " y: " << 0.0 << " z: " << 0.0 << " m: " << 0.0 << " theta: " << 0.0 << " phy: " << 0.0 << "\n";
    std::cout << "Final x: " << x << " y: " << y << " z: " << z << " m: " << m << " theta: " << theta << " phy: " << phy << " Gx: " << Gx << " Gy: " << Gy << " Gz: " << Gz << "\n";

    // set params
    std::vector<double> result_vec = {Gx, Gy, Gz, m, x, y, z, theta, phy};
    return result_vec;
}

// wrap C++ function with NumPy array IO
py::array_t<double> py_multiply(py::array_t<double, py::array::c_style | py::array::forcecast> array)
{
    // allocate std::vector (to pass to the C++ function)
    std::vector<double> array_vec(array.size());

    // copy py::array -> std::vector
    std::memcpy(array_vec.data(), array.data(), array.size() * sizeof(double));
    std::vector<double> result_vec(array.size());
    // call pure C++ function

    // std::vector<double> result_vec = multiply(array_vec);
    multiply2(array_vec, result_vec);

    // allocate py::array (to pass the result of the C++ function to Python)
    auto result = py::array_t<double>(array.size());
    auto result_buffer = result.request();
    double *result_ptr = (double *)result_buffer.ptr;

    // copy std::vector -> py::array
    std::memcpy(result_ptr, result_vec.data(), result_vec.size() * sizeof(double));

    return result;
}

py::array_t<double> py_solve_1mag(py::array_t<double, py::array::c_style | py::array::forcecast> readings, py::array_t<double, py::array::c_style | py::array::forcecast> pSensor, py::array_t<double, py::array::c_style | py::array::forcecast> init_param)
{
    // allocate std::vector (to pass to the C++ function)
    std::vector<double> readings_vec(readings.size());
    std::vector<double> pSensor_vec(pSensor.size());
    std::vector<double> init_param_vec(init_param.size());

    std::vector<double> result_vec(init_param.size());
    // copy py::array -> std::vector
    std::memcpy(readings_vec.data(), readings.data(), readings.size() * sizeof(double));
    std::memcpy(pSensor_vec.data(), pSensor.data(), pSensor.size() * sizeof(double));
    std::memcpy(init_param_vec.data(), init_param.data(), init_param.size() * sizeof(double));

    // call pure C++ function
    result_vec = solve_1mag(readings_vec, pSensor_vec, init_param_vec);
    // std::vector<double> result_vec = multiply(array_vec);
    // multiply2(array_vec, result_vec);

    // allocate py::array (to pass the result of the C++ function to Python)
    auto result = py::array_t<double>(init_param.size());
    auto result_buffer = result.request();
    double *result_ptr = (double *)result_buffer.ptr;

    // copy std::vector -> py::array
    std::memcpy(result_ptr, result_vec.data(), result_vec.size() * sizeof(double));

    return result;
}

PYBIND11_MODULE(cppsolver, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cppsolver

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def(
        "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("multiply", &py_multiply, "Convert all entries of an 1-D NumPy-array to int and multiply by 10");

    m.def("solve_1mag", &py_solve_1mag, "solve using the given parameters, sensor readings and sensor positions");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
