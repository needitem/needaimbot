























#pragma once
#ifndef IMGUI_DISABLE
#include "imgui.h"      


IMGUI_IMPL_API bool     ImGui_ImplGLUT_Init();
IMGUI_IMPL_API void     ImGui_ImplGLUT_InstallFuncs();
IMGUI_IMPL_API void     ImGui_ImplGLUT_Shutdown();
IMGUI_IMPL_API void     ImGui_ImplGLUT_NewFrame();




IMGUI_IMPL_API void     ImGui_ImplGLUT_ReshapeFunc(int w, int h);                           
IMGUI_IMPL_API void     ImGui_ImplGLUT_MotionFunc(int x, int y);                            
IMGUI_IMPL_API void     ImGui_ImplGLUT_MouseFunc(int button, int state, int x, int y);      
IMGUI_IMPL_API void     ImGui_ImplGLUT_MouseWheelFunc(int button, int dir, int x, int y);   
IMGUI_IMPL_API void     ImGui_ImplGLUT_KeyboardFunc(unsigned char c, int x, int y);         
IMGUI_IMPL_API void     ImGui_ImplGLUT_KeyboardUpFunc(unsigned char c, int x, int y);       
IMGUI_IMPL_API void     ImGui_ImplGLUT_SpecialFunc(int key, int x, int y);                  
IMGUI_IMPL_API void     ImGui_ImplGLUT_SpecialUpFunc(int key, int x, int y);                

#endif 

