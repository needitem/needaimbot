#ifndef SIMPLE_MATH_H
#define SIMPLE_MATH_H

#include <cmath>

struct Vec2f {
    float x, y;
    
    Vec2f() : x(0), y(0) {}
    Vec2f(float x_, float y_) : x(x_), y(y_) {}
    
    Vec2f operator+(const Vec2f& other) const {
        return Vec2f(x + other.x, y + other.y);
    }
    
    Vec2f operator-(const Vec2f& other) const {
        return Vec2f(x - other.x, y - other.y);
    }
    
    Vec2f operator*(float scalar) const {
        return Vec2f(x * scalar, y * scalar);
    }
    
    Vec2f& operator+=(const Vec2f& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
    
    Vec2f& operator-=(const Vec2f& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }
    
    float magnitude() const {
        return std::sqrt(x * x + y * y);
    }
    
    // Optimized: squared magnitude to avoid sqrt when only comparing distances
    float magnitudeSquared() const {
        return x * x + y * y;
    }
    
    // SIMD-friendly normalized operation
    Vec2f normalized() const {
        float mag = magnitude();
        return (mag > 1e-6f) ? Vec2f(x / mag, y / mag) : Vec2f(0, 0);
    }
    
    void reset() {
        x = y = 0;
    }
};

#endif // SIMPLE_MATH_H