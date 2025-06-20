




















#include "imgui.h"      
#ifndef IMGUI_DISABLE

#ifdef __OBJC__

@class NSEvent;
@class NSView;


IMGUI_IMPL_API bool     ImGui_ImplOSX_Init(NSView* _Nonnull view);
IMGUI_IMPL_API void     ImGui_ImplOSX_Shutdown();
IMGUI_IMPL_API void     ImGui_ImplOSX_NewFrame(NSView* _Nullable view);

#endif





#ifdef IMGUI_IMPL_METAL_CPP_EXTENSIONS

#ifndef __OBJC__


IMGUI_IMPL_API bool     ImGui_ImplOSX_Init(void* _Nonnull view);
IMGUI_IMPL_API void     ImGui_ImplOSX_Shutdown();
IMGUI_IMPL_API void     ImGui_ImplOSX_NewFrame(void* _Nullable view);

#endif
#endif

#endif 

