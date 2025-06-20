




















#pragma once
#ifndef IMGUI_DISABLE
#include "imgui.h"      

struct SDL_Renderer;


IMGUI_IMPL_API bool     ImGui_ImplSDLRenderer2_Init(SDL_Renderer* renderer);
IMGUI_IMPL_API void     ImGui_ImplSDLRenderer2_Shutdown();
IMGUI_IMPL_API void     ImGui_ImplSDLRenderer2_NewFrame();
IMGUI_IMPL_API void     ImGui_ImplSDLRenderer2_RenderDrawData(ImDrawData* draw_data, SDL_Renderer* renderer);


IMGUI_IMPL_API bool     ImGui_ImplSDLRenderer2_CreateFontsTexture();
IMGUI_IMPL_API void     ImGui_ImplSDLRenderer2_DestroyFontsTexture();
IMGUI_IMPL_API bool     ImGui_ImplSDLRenderer2_CreateDeviceObjects();
IMGUI_IMPL_API void     ImGui_ImplSDLRenderer2_DestroyDeviceObjects();

#endif 

