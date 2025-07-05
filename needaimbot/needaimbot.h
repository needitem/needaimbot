#ifndef NEEDAIMBOT_H
#define NEEDAIMBOT_H

#include "AppContext.h"

void add_to_history(std::vector<float>& history, float value, std::mutex& mtx, int max_size = 100);

#endif 
 
