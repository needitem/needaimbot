









#ifdef IMGUI_IMPLEMENTATION
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include "../../imgui.h"
#ifdef IMGUI_ENABLE_FREETYPE
#include "../../misc/freetype/imgui_freetype.h"
#endif

#ifdef IMGUI_IMPLEMENTATION
#include "../../imgui.cpp"
#include "../../imgui_demo.cpp"
#include "../../imgui_draw.cpp"
#include "../../imgui_tables.cpp"
#include "../../imgui_widgets.cpp"
#ifdef IMGUI_ENABLE_FREETYPE
#include "../../misc/freetype/imgui_freetype.cpp"
#endif
#endif

