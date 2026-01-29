#ifndef PREVIEW_WINDOW_H
#define PREVIEW_WINDOW_H

#include <atomic>

namespace PreviewWindow {

void Start();
void Stop();
void SetVisible(bool visible);
bool IsVisible();
bool IsRunning();

} // namespace PreviewWindow

#endif // PREVIEW_WINDOW_H
