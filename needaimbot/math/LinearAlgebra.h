#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

// Prevent Windows.h from defining min/max macros
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <cmath>
#include <cstring>
#include <algorithm>
#include <initializer_list>
#include <stdexcept>

namespace LA {

// 2D Vector class
class Vector2f {
private:
    float data[2];

public:
    // Constructors
    Vector2f() { data[0] = 0.0f; data[1] = 0.0f; }
    Vector2f(float x, float y) { data[0] = x; data[1] = y; }
    Vector2f(const Vector2f& other) { data[0] = other.data[0]; data[1] = other.data[1]; }

    // Static factory methods
    static Vector2f Zero() { return Vector2f(0.0f, 0.0f); }
    static Vector2f Ones() { return Vector2f(1.0f, 1.0f); }

    // Element access
    float& operator()(int i) { return data[i]; }
    const float& operator()(int i) const { return data[i]; }
    float& x() { return data[0]; }
    const float& x() const { return data[0]; }
    float& y() { return data[1]; }
    const float& y() const { return data[1]; }

    // Arithmetic operators
    Vector2f operator+(const Vector2f& other) const {
        return Vector2f(data[0] + other.data[0], data[1] + other.data[1]);
    }

    Vector2f operator-(const Vector2f& other) const {
        return Vector2f(data[0] - other.data[0], data[1] - other.data[1]);
    }

    Vector2f operator*(float scalar) const {
        return Vector2f(data[0] * scalar, data[1] * scalar);
    }

    Vector2f operator/(float scalar) const {
        return Vector2f(data[0] / scalar, data[1] / scalar);
    }

    Vector2f& operator+=(const Vector2f& other) {
        data[0] += other.data[0];
        data[1] += other.data[1];
        return *this;
    }

    Vector2f& operator-=(const Vector2f& other) {
        data[0] -= other.data[0];
        data[1] -= other.data[1];
        return *this;
    }

    Vector2f& operator*=(float scalar) {
        data[0] *= scalar;
        data[1] *= scalar;
        return *this;
    }

    Vector2f& operator=(const Vector2f& other) {
        data[0] = other.data[0];
        data[1] = other.data[1];
        return *this;
    }

    // Dot product
    float dot(const Vector2f& other) const {
        return data[0] * other.data[0] + data[1] * other.data[1];
    }

    // Norm
    float norm() const {
        return std::sqrt(data[0] * data[0] + data[1] * data[1]);
    }

    float squaredNorm() const {
        return data[0] * data[0] + data[1] * data[1];
    }

    // Normalization
    Vector2f normalized() const {
        float n = norm();
        if (n > 0) return *this / n;
        return *this;
    }

    void normalize() {
        float n = norm();
        if (n > 0) {
            data[0] /= n;
            data[1] /= n;
        }
    }
    
    // Friend function for scalar * vector multiplication
    friend Vector2f operator*(float scalar, const Vector2f& vec) {
        return Vector2f(scalar * vec.data[0], scalar * vec.data[1]);
    }
};

// Dynamic Vector class
class VectorXf {
private:
    float* data;
    int size_;

public:
    VectorXf() : data(nullptr), size_(0) {}
    
    explicit VectorXf(int size) : size_(size) {
        data = new float[size];
        std::memset(data, 0, size * sizeof(float));
    }

    VectorXf(const VectorXf& other) : size_(other.size_) {
        data = new float[size_];
        std::memcpy(data, other.data, size_ * sizeof(float));
    }

    VectorXf(std::initializer_list<float> list) : size_(static_cast<int>(list.size())) {
        data = new float[size_];
        std::copy(list.begin(), list.end(), data);
    }

    ~VectorXf() {
        delete[] data;
    }

    // Static factory methods
    static VectorXf Zero(int size) {
        return VectorXf(size);
    }

    static VectorXf Ones(int size) {
        VectorXf v(size);
        for (int i = 0; i < size; ++i) {
            v.data[i] = 1.0f;
        }
        return v;
    }

    // Size
    int size() const { return size_; }
    int rows() const { return size_; }

    // Element access
    float& operator()(int i) { return data[i]; }
    const float& operator()(int i) const { return data[i]; }
    float& operator[](int i) { return data[i]; }
    const float& operator[](int i) const { return data[i]; }

    // Assignment
    VectorXf& operator=(const VectorXf& other) {
        if (this != &other) {
            delete[] data;
            size_ = other.size_;
            data = new float[size_];
            std::memcpy(data, other.data, size_ * sizeof(float));
        }
        return *this;
    }

    // Arithmetic operators
    VectorXf operator+(const VectorXf& other) const {
        VectorXf result(size_);
        for (int i = 0; i < size_; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    VectorXf operator-(const VectorXf& other) const {
        VectorXf result(size_);
        for (int i = 0; i < size_; ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    VectorXf operator*(float scalar) const {
        VectorXf result(size_);
        for (int i = 0; i < size_; ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    VectorXf& operator+=(const VectorXf& other) {
        for (int i = 0; i < size_; ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    VectorXf& operator-=(const VectorXf& other) {
        for (int i = 0; i < size_; ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    // Set all elements to zero
    void setZero() {
        std::memset(data, 0, size_ * sizeof(float));
    }
};

// Dynamic Matrix class
class MatrixXf {
private:
    float* data;
    int rows_;
    int cols_;

public:
    MatrixXf() : data(nullptr), rows_(0), cols_(0) {}
    
    MatrixXf(int rows, int cols) : rows_(rows), cols_(cols) {
        data = new float[rows * cols];
        std::memset(data, 0, rows * cols * sizeof(float));
    }

    MatrixXf(const MatrixXf& other) : rows_(other.rows_), cols_(other.cols_) {
        int size = rows_ * cols_;
        data = new float[size];
        std::memcpy(data, other.data, size * sizeof(float));
    }

    ~MatrixXf() {
        delete[] data;
    }

    // Static factory methods
    static MatrixXf Zero(int rows, int cols) {
        return MatrixXf(rows, cols);
    }

    static MatrixXf Identity(int size) {
        MatrixXf m(size, size);
        for (int i = 0; i < size; ++i) {
            m(i, i) = 1.0f;
        }
        return m;
    }

    static MatrixXf Ones(int rows, int cols) {
        MatrixXf m(rows, cols);
        for (int i = 0; i < rows * cols; ++i) {
            m.data[i] = 1.0f;
        }
        return m;
    }

    // Size
    int rows() const { return rows_; }
    int cols() const { return cols_; }

    // Element access
    float& operator()(int row, int col) {
        return data[row * cols_ + col];
    }

    const float& operator()(int row, int col) const {
        return data[row * cols_ + col];
    }

    // Assignment
    MatrixXf& operator=(const MatrixXf& other) {
        if (this != &other) {
            delete[] data;
            rows_ = other.rows_;
            cols_ = other.cols_;
            int size = rows_ * cols_;
            data = new float[size];
            std::memcpy(data, other.data, size * sizeof(float));
        }
        return *this;
    }

    // Matrix operations
    MatrixXf operator+(const MatrixXf& other) const {
        MatrixXf result(rows_, cols_);
        int size = rows_ * cols_;
        for (int i = 0; i < size; ++i) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    MatrixXf operator-(const MatrixXf& other) const {
        MatrixXf result(rows_, cols_);
        int size = rows_ * cols_;
        for (int i = 0; i < size; ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    MatrixXf operator*(const MatrixXf& other) const {
        if (cols_ != other.rows_) {
            throw std::runtime_error("Matrix dimensions don't match for multiplication");
        }
        
        MatrixXf result(rows_, other.cols_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < other.cols_; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < cols_; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    VectorXf operator*(const VectorXf& vec) const {
        if (cols_ != vec.size()) {
            throw std::runtime_error("Matrix-vector dimensions don't match");
        }
        
        VectorXf result(rows_);
        for (int i = 0; i < rows_; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < cols_; ++j) {
                sum += (*this)(i, j) * vec(j);
            }
            result(i) = sum;
        }
        return result;
    }

    MatrixXf operator*(float scalar) const {
        MatrixXf result(rows_, cols_);
        int size = rows_ * cols_;
        for (int i = 0; i < size; ++i) {
            result.data[i] = data[i] * scalar;
        }
        return result;
    }

    MatrixXf& operator+=(const MatrixXf& other) {
        int size = rows_ * cols_;
        for (int i = 0; i < size; ++i) {
            data[i] += other.data[i];
        }
        return *this;
    }

    MatrixXf& operator-=(const MatrixXf& other) {
        int size = rows_ * cols_;
        for (int i = 0; i < size; ++i) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    // Transpose
    MatrixXf transpose() const {
        MatrixXf result(cols_, rows_);
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    // Inverse (only for small matrices using Gauss-Jordan elimination)
    MatrixXf inverse() const {
        if (rows_ != cols_) {
            throw std::runtime_error("Cannot invert non-square matrix");
        }
        
        int n = rows_;
        MatrixXf aug(n, 2 * n);
        
        // Create augmented matrix [A | I]
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                aug(i, j) = (*this)(i, j);
                aug(i, j + n) = (i == j) ? 1.0f : 0.0f;
            }
        }
        
        // Gauss-Jordan elimination
        for (int i = 0; i < n; ++i) {
            // Find pivot
            float pivot = aug(i, i);
            if (std::abs(pivot) < 1e-10) {
                throw std::runtime_error("Matrix is singular");
            }
            
            // Scale row
            for (int j = 0; j < 2 * n; ++j) {
                aug(i, j) /= pivot;
            }
            
            // Eliminate column
            for (int k = 0; k < n; ++k) {
                if (k != i) {
                    float factor = aug(k, i);
                    for (int j = 0; j < 2 * n; ++j) {
                        aug(k, j) -= factor * aug(i, j);
                    }
                }
            }
        }
        
        // Extract inverse from augmented matrix
        MatrixXf result(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                result(i, j) = aug(i, j + n);
            }
        }
        
        return result;
    }

    // Set all elements to zero
    void setZero() {
        std::memset(data, 0, rows_ * cols_ * sizeof(float));
    }

    // Set to identity matrix
    void setIdentity() {
        setZero();
        int min_dim = (std::min)(rows_, cols_);
        for (int i = 0; i < min_dim; ++i) {
            (*this)(i, i) = 1.0f;
        }
    }
};

} // namespace LA

#endif // LINEAR_ALGEBRA_H