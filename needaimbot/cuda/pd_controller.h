// PD Controller Header for CPU-side usage
#pragma once

namespace needaimbot {
namespace cuda {

// Forward declaration of reset function
void resetAllPDStates();
void resetTargetPDState(int target_id);

} // namespace cuda
} // namespace needaimbot