



















#pragma once
#include "imgui.h"      
#ifndef IMGUI_DISABLE

struct ALLEGRO_DISPLAY;
union ALLEGRO_EVENT;


IMGUI_IMPL_API bool     ImGui_ImplAllegro5_Init(ALLEGRO_DISPLAY* display);
IMGUI_IMPL_API void     ImGui_ImplAllegro5_Shutdown();
IMGUI_IMPL_API void     ImGui_ImplAllegro5_NewFrame();
IMGUI_IMPL_API void     ImGui_ImplAllegro5_RenderDrawData(ImDrawData* draw_data);
IMGUI_IMPL_API bool     ImGui_ImplAllegro5_ProcessEvent(ALLEGRO_EVENT* event);


IMGUI_IMPL_API bool     ImGui_ImplAllegro5_CreateDeviceObjects();
IMGUI_IMPL_API void     ImGui_ImplAllegro5_InvalidateDeviceObjects();

#endif 

