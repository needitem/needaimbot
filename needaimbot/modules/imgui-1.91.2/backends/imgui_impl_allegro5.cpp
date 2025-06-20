






















































#include "imgui.h"
#ifndef IMGUI_DISABLE
#include "imgui_impl_allegro5.h"
#include <stdint.h>     
#include <cstring>      


#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#ifdef _WIN32
#include <allegro5/allegro_windows.h>
#endif
#define ALLEGRO_HAS_CLIPBOARD           ((ALLEGRO_VERSION_INT & ~ALLEGRO_UNSTABLE_BIT) >= ((5 << 24) | (1 << 16) | (12 << 8))) 
#define ALLEGRO_HAS_DRAW_INDEXED_PRIM   ((ALLEGRO_VERSION_INT & ~ALLEGRO_UNSTABLE_BIT) >= ((5 << 24) | (2 << 16) | ( 5 << 8))) 


#ifdef _MSC_VER
#pragma warning (disable: 4127) 
#endif

struct ImDrawVertAllegro
{
    ImVec2          pos;
    ImVec2          uv;
    ALLEGRO_COLOR   col;
};





#define DRAW_VERT_IMGUI_TO_ALLEGRO(DST, SRC)  { (DST)->pos = (SRC)->pos; (DST)->uv = (SRC)->uv; unsigned char* c = (unsigned char*)&(SRC)->col; (DST)->col = al_map_rgba(c[0], c[1], c[2], c[3]); }


struct ImGui_ImplAllegro5_Data
{
    ALLEGRO_DISPLAY*            Display;
    ALLEGRO_BITMAP*             Texture;
    double                      Time;
    ALLEGRO_MOUSE_CURSOR*       MouseCursorInvisible;
    ALLEGRO_VERTEX_DECL*        VertexDecl;
    char*                       ClipboardTextData;

    ImVector<ImDrawVertAllegro> BufVertices;
    ImVector<int>               BufIndices;

    ImGui_ImplAllegro5_Data()   { memset((void*)this, 0, sizeof(*this)); }
};




static ImGui_ImplAllegro5_Data* ImGui_ImplAllegro5_GetBackendData()     { return ImGui::GetCurrentContext() ? (ImGui_ImplAllegro5_Data*)ImGui::GetIO().BackendPlatformUserData : nullptr; }

static void ImGui_ImplAllegro5_SetupRenderState(ImDrawData* draw_data)
{
    
    al_set_separate_blender(ALLEGRO_ADD, ALLEGRO_ALPHA, ALLEGRO_INVERSE_ALPHA, ALLEGRO_ADD, ALLEGRO_ONE, ALLEGRO_INVERSE_ALPHA);

    
    
    {
        float L = draw_data->DisplayPos.x;
        float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
        float T = draw_data->DisplayPos.y;
        float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
        ALLEGRO_TRANSFORM transform;
        al_identity_transform(&transform);
        al_use_transform(&transform);
        al_orthographic_transform(&transform, L, T, 1.0f, R, B, -1.0f);
        al_use_projection_transform(&transform);
    }
}


void ImGui_ImplAllegro5_RenderDrawData(ImDrawData* draw_data)
{
    
    if (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f)
        return;

    
    ImGui_ImplAllegro5_Data* bd = ImGui_ImplAllegro5_GetBackendData();
    ALLEGRO_TRANSFORM last_transform = *al_get_current_transform();
    ALLEGRO_TRANSFORM last_projection_transform = *al_get_current_projection_transform();
    int last_clip_x, last_clip_y, last_clip_w, last_clip_h;
    al_get_clipping_rectangle(&last_clip_x, &last_clip_y, &last_clip_w, &last_clip_h);
    int last_blender_op, last_blender_src, last_blender_dst;
    al_get_blender(&last_blender_op, &last_blender_src, &last_blender_dst);

    
    ImGui_ImplAllegro5_SetupRenderState(draw_data);

    
    for (int n = 0; n < draw_data->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list = draw_data->CmdLists[n];

        ImVector<ImDrawVertAllegro>& vertices = bd->BufVertices;
#if ALLEGRO_HAS_DRAW_INDEXED_PRIM
        vertices.resize(cmd_list->VtxBuffer.Size);
        for (int i = 0; i < cmd_list->VtxBuffer.Size; i++)
        {
            const ImDrawVert* src_v = &cmd_list->VtxBuffer[i];
            ImDrawVertAllegro* dst_v = &vertices[i];
            DRAW_VERT_IMGUI_TO_ALLEGRO(dst_v, src_v);
        }
        const int* indices = nullptr;
        if (sizeof(ImDrawIdx) == 2)
        {
            
            
            
            bd->BufIndices.resize(cmd_list->IdxBuffer.Size);
            for (int i = 0; i < cmd_list->IdxBuffer.Size; ++i)
                bd->BufIndices[i] = (int)cmd_list->IdxBuffer.Data[i];
            indices = bd->BufIndices.Data;
        }
        else if (sizeof(ImDrawIdx) == 4)
        {
            indices = (const int*)cmd_list->IdxBuffer.Data;
        }
#else
        
        vertices.resize(cmd_list->IdxBuffer.Size);
        for (int i = 0; i < cmd_list->IdxBuffer.Size; i++)
        {
            const ImDrawVert* src_v = &cmd_list->VtxBuffer[cmd_list->IdxBuffer[i]];
            ImDrawVertAllegro* dst_v = &vertices[i];
            DRAW_VERT_IMGUI_TO_ALLEGRO(dst_v, src_v);
        }
#endif

        
        ImVec2 clip_off = draw_data->DisplayPos;
        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++)
        {
            const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
            if (pcmd->UserCallback)
            {
                
                
                if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
                    ImGui_ImplAllegro5_SetupRenderState(draw_data);
                else
                    pcmd->UserCallback(cmd_list, pcmd);
            }
            else
            {
                
                ImVec2 clip_min(pcmd->ClipRect.x - clip_off.x, pcmd->ClipRect.y - clip_off.y);
                ImVec2 clip_max(pcmd->ClipRect.z - clip_off.x, pcmd->ClipRect.w - clip_off.y);
                if (clip_max.x <= clip_min.x || clip_max.y <= clip_min.y)
                    continue;

                
                ALLEGRO_BITMAP* texture = (ALLEGRO_BITMAP*)pcmd->GetTexID();
                al_set_clipping_rectangle(clip_min.x, clip_min.y, clip_max.x - clip_min.x, clip_max.y - clip_min.y);
#if ALLEGRO_HAS_DRAW_INDEXED_PRIM
                al_draw_indexed_prim(&vertices[0], bd->VertexDecl, texture, &indices[pcmd->IdxOffset], pcmd->ElemCount, ALLEGRO_PRIM_TRIANGLE_LIST);
#else
                al_draw_prim(&vertices[0], bd->VertexDecl, texture, pcmd->IdxOffset, pcmd->IdxOffset + pcmd->ElemCount, ALLEGRO_PRIM_TRIANGLE_LIST);
#endif
            }
        }
    }

    
    al_set_blender(last_blender_op, last_blender_src, last_blender_dst);
    al_set_clipping_rectangle(last_clip_x, last_clip_y, last_clip_w, last_clip_h);
    al_use_transform(&last_transform);
    al_use_projection_transform(&last_projection_transform);
}

bool ImGui_ImplAllegro5_CreateDeviceObjects()
{
    
    ImGui_ImplAllegro5_Data* bd = ImGui_ImplAllegro5_GetBackendData();
    ImGuiIO& io = ImGui::GetIO();
    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    
    
    int flags = al_get_new_bitmap_flags();
    int fmt = al_get_new_bitmap_format();
    al_set_new_bitmap_flags(ALLEGRO_MEMORY_BITMAP | ALLEGRO_MIN_LINEAR | ALLEGRO_MAG_LINEAR);
    al_set_new_bitmap_format(ALLEGRO_PIXEL_FORMAT_ABGR_8888_LE);
    ALLEGRO_BITMAP* img = al_create_bitmap(width, height);
    al_set_new_bitmap_flags(flags);
    al_set_new_bitmap_format(fmt);
    if (!img)
        return false;

    ALLEGRO_LOCKED_REGION* locked_img = al_lock_bitmap(img, al_get_bitmap_format(img), ALLEGRO_LOCK_WRITEONLY);
    if (!locked_img)
    {
        al_destroy_bitmap(img);
        return false;
    }
    memcpy(locked_img->data, pixels, sizeof(int) * width * height);
    al_unlock_bitmap(img);

    
    ALLEGRO_BITMAP* cloned_img = al_clone_bitmap(img);
    al_destroy_bitmap(img);
    if (!cloned_img)
        return false;

    
    io.Fonts->SetTexID((ImTextureID)(intptr_t)cloned_img);
    bd->Texture = cloned_img;

    
    
    ALLEGRO_BITMAP* mouse_cursor = al_create_bitmap(8, 8);
    bd->MouseCursorInvisible = al_create_mouse_cursor(mouse_cursor, 0, 0);
    al_destroy_bitmap(mouse_cursor);

    return true;
}

void ImGui_ImplAllegro5_InvalidateDeviceObjects()
{
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplAllegro5_Data* bd = ImGui_ImplAllegro5_GetBackendData();
    if (bd->Texture)
    {
        io.Fonts->SetTexID(0);
        al_destroy_bitmap(bd->Texture);
        bd->Texture = nullptr;
    }
    if (bd->MouseCursorInvisible)
    {
        al_destroy_mouse_cursor(bd->MouseCursorInvisible);
        bd->MouseCursorInvisible = nullptr;
    }
}

#if ALLEGRO_HAS_CLIPBOARD
static const char* ImGui_ImplAllegro5_GetClipboardText(ImGuiContext*)
{
    ImGui_ImplAllegro5_Data* bd = ImGui_ImplAllegro5_GetBackendData();
    if (bd->ClipboardTextData)
        al_free(bd->ClipboardTextData);
    bd->ClipboardTextData = al_get_clipboard_text(bd->Display);
    return bd->ClipboardTextData;
}

static void ImGui_ImplAllegro5_SetClipboardText(ImGuiContext*, const char* text)
{
    ImGui_ImplAllegro5_Data* bd = ImGui_ImplAllegro5_GetBackendData();
    al_set_clipboard_text(bd->Display, text);
}
#endif


ImGuiKey ImGui_ImplAllegro5_KeyCodeToImGuiKey(int key_code)
{
    switch (key_code)
    {
        case ALLEGRO_KEY_TAB: return ImGuiKey_Tab;
        case ALLEGRO_KEY_LEFT: return ImGuiKey_LeftArrow;
        case ALLEGRO_KEY_RIGHT: return ImGuiKey_RightArrow;
        case ALLEGRO_KEY_UP: return ImGuiKey_UpArrow;
        case ALLEGRO_KEY_DOWN: return ImGuiKey_DownArrow;
        case ALLEGRO_KEY_PGUP: return ImGuiKey_PageUp;
        case ALLEGRO_KEY_PGDN: return ImGuiKey_PageDown;
        case ALLEGRO_KEY_HOME: return ImGuiKey_Home;
        case ALLEGRO_KEY_END: return ImGuiKey_End;
        case ALLEGRO_KEY_INSERT: return ImGuiKey_Insert;
        case ALLEGRO_KEY_DELETE: return ImGuiKey_Delete;
        case ALLEGRO_KEY_BACKSPACE: return ImGuiKey_Backspace;
        case ALLEGRO_KEY_SPACE: return ImGuiKey_Space;
        case ALLEGRO_KEY_ENTER: return ImGuiKey_Enter;
        case ALLEGRO_KEY_ESCAPE: return ImGuiKey_Escape;
        case ALLEGRO_KEY_QUOTE: return ImGuiKey_Apostrophe;
        case ALLEGRO_KEY_COMMA: return ImGuiKey_Comma;
        case ALLEGRO_KEY_MINUS: return ImGuiKey_Minus;
        case ALLEGRO_KEY_FULLSTOP: return ImGuiKey_Period;
        case ALLEGRO_KEY_SLASH: return ImGuiKey_Slash;
        case ALLEGRO_KEY_SEMICOLON: return ImGuiKey_Semicolon;
        case ALLEGRO_KEY_EQUALS: return ImGuiKey_Equal;
        case ALLEGRO_KEY_OPENBRACE: return ImGuiKey_LeftBracket;
        case ALLEGRO_KEY_BACKSLASH: return ImGuiKey_Backslash;
        case ALLEGRO_KEY_CLOSEBRACE: return ImGuiKey_RightBracket;
        case ALLEGRO_KEY_TILDE: return ImGuiKey_GraveAccent;
        case ALLEGRO_KEY_CAPSLOCK: return ImGuiKey_CapsLock;
        case ALLEGRO_KEY_SCROLLLOCK: return ImGuiKey_ScrollLock;
        case ALLEGRO_KEY_NUMLOCK: return ImGuiKey_NumLock;
        case ALLEGRO_KEY_PRINTSCREEN: return ImGuiKey_PrintScreen;
        case ALLEGRO_KEY_PAUSE: return ImGuiKey_Pause;
        case ALLEGRO_KEY_PAD_0: return ImGuiKey_Keypad0;
        case ALLEGRO_KEY_PAD_1: return ImGuiKey_Keypad1;
        case ALLEGRO_KEY_PAD_2: return ImGuiKey_Keypad2;
        case ALLEGRO_KEY_PAD_3: return ImGuiKey_Keypad3;
        case ALLEGRO_KEY_PAD_4: return ImGuiKey_Keypad4;
        case ALLEGRO_KEY_PAD_5: return ImGuiKey_Keypad5;
        case ALLEGRO_KEY_PAD_6: return ImGuiKey_Keypad6;
        case ALLEGRO_KEY_PAD_7: return ImGuiKey_Keypad7;
        case ALLEGRO_KEY_PAD_8: return ImGuiKey_Keypad8;
        case ALLEGRO_KEY_PAD_9: return ImGuiKey_Keypad9;
        case ALLEGRO_KEY_PAD_DELETE: return ImGuiKey_KeypadDecimal;
        case ALLEGRO_KEY_PAD_SLASH: return ImGuiKey_KeypadDivide;
        case ALLEGRO_KEY_PAD_ASTERISK: return ImGuiKey_KeypadMultiply;
        case ALLEGRO_KEY_PAD_MINUS: return ImGuiKey_KeypadSubtract;
        case ALLEGRO_KEY_PAD_PLUS: return ImGuiKey_KeypadAdd;
        case ALLEGRO_KEY_PAD_ENTER: return ImGuiKey_KeypadEnter;
        case ALLEGRO_KEY_PAD_EQUALS: return ImGuiKey_KeypadEqual;
        case ALLEGRO_KEY_LCTRL: return ImGuiKey_LeftCtrl;
        case ALLEGRO_KEY_LSHIFT: return ImGuiKey_LeftShift;
        case ALLEGRO_KEY_ALT: return ImGuiKey_LeftAlt;
        case ALLEGRO_KEY_LWIN: return ImGuiKey_LeftSuper;
        case ALLEGRO_KEY_RCTRL: return ImGuiKey_RightCtrl;
        case ALLEGRO_KEY_RSHIFT: return ImGuiKey_RightShift;
        case ALLEGRO_KEY_ALTGR: return ImGuiKey_RightAlt;
        case ALLEGRO_KEY_RWIN: return ImGuiKey_RightSuper;
        case ALLEGRO_KEY_MENU: return ImGuiKey_Menu;
        case ALLEGRO_KEY_0: return ImGuiKey_0;
        case ALLEGRO_KEY_1: return ImGuiKey_1;
        case ALLEGRO_KEY_2: return ImGuiKey_2;
        case ALLEGRO_KEY_3: return ImGuiKey_3;
        case ALLEGRO_KEY_4: return ImGuiKey_4;
        case ALLEGRO_KEY_5: return ImGuiKey_5;
        case ALLEGRO_KEY_6: return ImGuiKey_6;
        case ALLEGRO_KEY_7: return ImGuiKey_7;
        case ALLEGRO_KEY_8: return ImGuiKey_8;
        case ALLEGRO_KEY_9: return ImGuiKey_9;
        case ALLEGRO_KEY_A: return ImGuiKey_A;
        case ALLEGRO_KEY_B: return ImGuiKey_B;
        case ALLEGRO_KEY_C: return ImGuiKey_C;
        case ALLEGRO_KEY_D: return ImGuiKey_D;
        case ALLEGRO_KEY_E: return ImGuiKey_E;
        case ALLEGRO_KEY_F: return ImGuiKey_F;
        case ALLEGRO_KEY_G: return ImGuiKey_G;
        case ALLEGRO_KEY_H: return ImGuiKey_H;
        case ALLEGRO_KEY_I: return ImGuiKey_I;
        case ALLEGRO_KEY_J: return ImGuiKey_J;
        case ALLEGRO_KEY_K: return ImGuiKey_K;
        case ALLEGRO_KEY_L: return ImGuiKey_L;
        case ALLEGRO_KEY_M: return ImGuiKey_M;
        case ALLEGRO_KEY_N: return ImGuiKey_N;
        case ALLEGRO_KEY_O: return ImGuiKey_O;
        case ALLEGRO_KEY_P: return ImGuiKey_P;
        case ALLEGRO_KEY_Q: return ImGuiKey_Q;
        case ALLEGRO_KEY_R: return ImGuiKey_R;
        case ALLEGRO_KEY_S: return ImGuiKey_S;
        case ALLEGRO_KEY_T: return ImGuiKey_T;
        case ALLEGRO_KEY_U: return ImGuiKey_U;
        case ALLEGRO_KEY_V: return ImGuiKey_V;
        case ALLEGRO_KEY_W: return ImGuiKey_W;
        case ALLEGRO_KEY_X: return ImGuiKey_X;
        case ALLEGRO_KEY_Y: return ImGuiKey_Y;
        case ALLEGRO_KEY_Z: return ImGuiKey_Z;
        case ALLEGRO_KEY_F1: return ImGuiKey_F1;
        case ALLEGRO_KEY_F2: return ImGuiKey_F2;
        case ALLEGRO_KEY_F3: return ImGuiKey_F3;
        case ALLEGRO_KEY_F4: return ImGuiKey_F4;
        case ALLEGRO_KEY_F5: return ImGuiKey_F5;
        case ALLEGRO_KEY_F6: return ImGuiKey_F6;
        case ALLEGRO_KEY_F7: return ImGuiKey_F7;
        case ALLEGRO_KEY_F8: return ImGuiKey_F8;
        case ALLEGRO_KEY_F9: return ImGuiKey_F9;
        case ALLEGRO_KEY_F10: return ImGuiKey_F10;
        case ALLEGRO_KEY_F11: return ImGuiKey_F11;
        case ALLEGRO_KEY_F12: return ImGuiKey_F12;
        default: return ImGuiKey_None;
    }
}

bool ImGui_ImplAllegro5_Init(ALLEGRO_DISPLAY* display)
{
    ImGuiIO& io = ImGui::GetIO();
    IMGUI_CHECKVERSION();
    IM_ASSERT(io.BackendPlatformUserData == nullptr && "Already initialized a platform backend!");

    
    ImGui_ImplAllegro5_Data* bd = IM_NEW(ImGui_ImplAllegro5_Data)();
    io.BackendPlatformUserData = (void*)bd;
    io.BackendPlatformName = io.BackendRendererName = "imgui_impl_allegro5";
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;       

    bd->Display = display;

    
    
    
    ALLEGRO_VERTEX_ELEMENT elems[] =
    {
        { ALLEGRO_PRIM_POSITION, ALLEGRO_PRIM_FLOAT_2, offsetof(ImDrawVertAllegro, pos) },
        { ALLEGRO_PRIM_TEX_COORD, ALLEGRO_PRIM_FLOAT_2, offsetof(ImDrawVertAllegro, uv) },
        { ALLEGRO_PRIM_COLOR_ATTR, 0, offsetof(ImDrawVertAllegro, col) },
        { 0, 0, 0 }
    };
    bd->VertexDecl = al_create_vertex_decl(elems, sizeof(ImDrawVertAllegro));

#if ALLEGRO_HAS_CLIPBOARD
    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
    platform_io.Platform_SetClipboardTextFn = ImGui_ImplAllegro5_SetClipboardText;
    platform_io.Platform_GetClipboardTextFn = ImGui_ImplAllegro5_GetClipboardText;
#endif

    return true;
}

void ImGui_ImplAllegro5_Shutdown()
{
    ImGui_ImplAllegro5_Data* bd = ImGui_ImplAllegro5_GetBackendData();
    IM_ASSERT(bd != nullptr && "No platform backend to shutdown, or already shutdown?");
    ImGuiIO& io = ImGui::GetIO();

    ImGui_ImplAllegro5_InvalidateDeviceObjects();
    if (bd->VertexDecl)
        al_destroy_vertex_decl(bd->VertexDecl);
    if (bd->ClipboardTextData)
        al_free(bd->ClipboardTextData);

    io.BackendPlatformName = io.BackendRendererName = nullptr;
    io.BackendPlatformUserData = nullptr;
    io.BackendFlags &= ~ImGuiBackendFlags_HasMouseCursors;
    IM_DELETE(bd);
}


static void ImGui_ImplAllegro5_UpdateKeyModifiers()
{
    ImGuiIO& io = ImGui::GetIO();
    ALLEGRO_KEYBOARD_STATE keys;
    al_get_keyboard_state(&keys);
    io.AddKeyEvent(ImGuiMod_Ctrl, al_key_down(&keys, ALLEGRO_KEY_LCTRL) || al_key_down(&keys, ALLEGRO_KEY_RCTRL));
    io.AddKeyEvent(ImGuiMod_Shift, al_key_down(&keys, ALLEGRO_KEY_LSHIFT) || al_key_down(&keys, ALLEGRO_KEY_RSHIFT));
    io.AddKeyEvent(ImGuiMod_Alt, al_key_down(&keys, ALLEGRO_KEY_ALT) || al_key_down(&keys, ALLEGRO_KEY_ALTGR));
    io.AddKeyEvent(ImGuiMod_Super, al_key_down(&keys, ALLEGRO_KEY_LWIN) || al_key_down(&keys, ALLEGRO_KEY_RWIN));
}





bool ImGui_ImplAllegro5_ProcessEvent(ALLEGRO_EVENT* ev)
{
    ImGui_ImplAllegro5_Data* bd = ImGui_ImplAllegro5_GetBackendData();
    IM_ASSERT(bd != nullptr && "Context or backend not initialized! Did you call ImGui_ImplAllegro5_Init()?");
    ImGuiIO& io = ImGui::GetIO();

    switch (ev->type)
    {
    case ALLEGRO_EVENT_MOUSE_AXES:
        if (ev->mouse.display == bd->Display)
        {
            io.AddMousePosEvent(ev->mouse.x, ev->mouse.y);
            io.AddMouseWheelEvent(-ev->mouse.dw, ev->mouse.dz);
        }
        return true;
    case ALLEGRO_EVENT_MOUSE_BUTTON_DOWN:
    case ALLEGRO_EVENT_MOUSE_BUTTON_UP:
        if (ev->mouse.display == bd->Display && ev->mouse.button > 0 && ev->mouse.button <= 5)
            io.AddMouseButtonEvent(ev->mouse.button - 1, ev->type == ALLEGRO_EVENT_MOUSE_BUTTON_DOWN);
        return true;
    case ALLEGRO_EVENT_TOUCH_MOVE:
        if (ev->touch.display == bd->Display)
            io.AddMousePosEvent(ev->touch.x, ev->touch.y);
        return true;
    case ALLEGRO_EVENT_TOUCH_BEGIN:
    case ALLEGRO_EVENT_TOUCH_END:
    case ALLEGRO_EVENT_TOUCH_CANCEL:
        if (ev->touch.display == bd->Display && ev->touch.primary)
            io.AddMouseButtonEvent(0, ev->type == ALLEGRO_EVENT_TOUCH_BEGIN);
        return true;
    case ALLEGRO_EVENT_MOUSE_LEAVE_DISPLAY:
        if (ev->mouse.display == bd->Display)
            io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
        return true;
    case ALLEGRO_EVENT_KEY_CHAR:
        if (ev->keyboard.display == bd->Display)
            if (ev->keyboard.unichar != 0)
                io.AddInputCharacter((unsigned int)ev->keyboard.unichar);
        return true;
    case ALLEGRO_EVENT_KEY_DOWN:
    case ALLEGRO_EVENT_KEY_UP:
        if (ev->keyboard.display == bd->Display)
        {
            ImGui_ImplAllegro5_UpdateKeyModifiers();
            ImGuiKey key = ImGui_ImplAllegro5_KeyCodeToImGuiKey(ev->keyboard.keycode);
            io.AddKeyEvent(key, (ev->type == ALLEGRO_EVENT_KEY_DOWN));
            io.SetKeyEventNativeData(key, ev->keyboard.keycode, -1); 
        }
        return true;
    case ALLEGRO_EVENT_DISPLAY_SWITCH_OUT:
        if (ev->display.source == bd->Display)
            io.AddFocusEvent(false);
        return true;
    case ALLEGRO_EVENT_DISPLAY_SWITCH_IN:
        if (ev->display.source == bd->Display)
        {
            io.AddFocusEvent(true);
#if defined(ALLEGRO_UNSTABLE)
            al_clear_keyboard_state(bd->Display);
#endif
        }
        return true;
    }
    return false;
}

static void ImGui_ImplAllegro5_UpdateMouseCursor()
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_NoMouseCursorChange)
        return;

    ImGui_ImplAllegro5_Data* bd = ImGui_ImplAllegro5_GetBackendData();
    ImGuiMouseCursor imgui_cursor = ImGui::GetMouseCursor();
    if (io.MouseDrawCursor || imgui_cursor == ImGuiMouseCursor_None)
    {
        
        al_set_mouse_cursor(bd->Display, bd->MouseCursorInvisible);
    }
    else
    {
        ALLEGRO_SYSTEM_MOUSE_CURSOR cursor_id = ALLEGRO_SYSTEM_MOUSE_CURSOR_DEFAULT;
        switch (imgui_cursor)
        {
        case ImGuiMouseCursor_TextInput:    cursor_id = ALLEGRO_SYSTEM_MOUSE_CURSOR_EDIT; break;
        case ImGuiMouseCursor_ResizeAll:    cursor_id = ALLEGRO_SYSTEM_MOUSE_CURSOR_MOVE; break;
        case ImGuiMouseCursor_ResizeNS:     cursor_id = ALLEGRO_SYSTEM_MOUSE_CURSOR_RESIZE_N; break;
        case ImGuiMouseCursor_ResizeEW:     cursor_id = ALLEGRO_SYSTEM_MOUSE_CURSOR_RESIZE_E; break;
        case ImGuiMouseCursor_ResizeNESW:   cursor_id = ALLEGRO_SYSTEM_MOUSE_CURSOR_RESIZE_NE; break;
        case ImGuiMouseCursor_ResizeNWSE:   cursor_id = ALLEGRO_SYSTEM_MOUSE_CURSOR_RESIZE_NW; break;
        case ImGuiMouseCursor_NotAllowed:   cursor_id = ALLEGRO_SYSTEM_MOUSE_CURSOR_UNAVAILABLE; break;
        }
        al_set_system_mouse_cursor(bd->Display, cursor_id);
    }
}

void ImGui_ImplAllegro5_NewFrame()
{
    ImGui_ImplAllegro5_Data* bd = ImGui_ImplAllegro5_GetBackendData();
    IM_ASSERT(bd != nullptr && "Context or backend not initialized! Did you call ImGui_ImplAllegro5_Init()?");

    if (!bd->Texture)
        ImGui_ImplAllegro5_CreateDeviceObjects();

    ImGuiIO& io = ImGui::GetIO();

    
    int w, h;
    w = al_get_display_width(bd->Display);
    h = al_get_display_height(bd->Display);
    io.DisplaySize = ImVec2((float)w, (float)h);

    
    double current_time = al_get_time();
    io.DeltaTime = bd->Time > 0.0 ? (float)(current_time - bd->Time) : (float)(1.0f / 60.0f);
    bd->Time = current_time;

    
    ImGui_ImplAllegro5_UpdateMouseCursor();
}



#endif 

