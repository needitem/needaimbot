






















#pragma once
#include "imgui.h"      
#ifndef IMGUI_DISABLE

struct SDL_Renderer;


IMGUI_IMPL_API bool     ImGui_ImplSDLRenderer3_Init(SDL_Renderer* renderer);
IMGUI_IMPL_API void     ImGui_ImplSDLRenderer3_Shutdown();
IMGUI_IMPL_API void     ImGui_ImplSDLRenderer3_NewFrame();
IMGUI_IMPL_API void     ImGui_ImplSDLRenderer3_RenderDrawData(ImDrawData* draw_data, SDL_Renderer* renderer);


IMGUI_IMPL_API bool     ImGui_ImplSDLRenderer3_CreateFontsTexture();
IMGUI_IMPL_API void     ImGui_ImplSDLRenderer3_DestroyFontsTexture();
IMGUI_IMPL_API bool     ImGui_ImplSDLRenderer3_CreateDeviceObjects();
IMGUI_IMPL_API void     ImGui_ImplSDLRenderer3_DestroyDeviceObjects();

#endif 

