#ifndef IPREDICTOR_H
#define IPREDICTOR_H

#include <chrono>
#include <string> 
#include <vector> 


struct Point2D {
    float x;
    float y;
};


class IPredictor {
public:
    
    virtual ~IPredictor() = default;

    
    

    
    virtual void update(const Point2D& position, std::chrono::steady_clock::time_point timestamp) = 0;

    
    virtual Point2D predict() const = 0;

    
    virtual void reset() = 0;
};

#endif 
