import time
import threading
import pynput.mouse as mouse
import pynput.keyboard as keyboard

class AutoClicker:
    def __init__(self):
        self.clicking = False
        self.running = True
        self.click_delay = 0.05  # 클릭 간격 (초)
        self.mouse_controller = mouse.Controller()
        self.mouse_listener = None
        self.keyboard_listener = None
        
    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            if pressed:
                self.clicking = True
                threading.Thread(target=self.auto_click, daemon=True).start()
            else:
                self.clicking = False
                
    def auto_click(self):
        while self.clicking and self.running:
            self.mouse_controller.click(mouse.Button.left)
            time.sleep(self.click_delay)
            
    def on_key(self, key):
        try:
            if key == keyboard.Key.f1:  # F1으로 속도 증가
                self.click_delay = max(0.01, self.click_delay - 0.01)
                print(f"클릭 속도 증가: {1/self.click_delay:.1f} CPS")
            elif key == keyboard.Key.f2:  # F2로 속도 감소
                self.click_delay = min(0.5, self.click_delay + 0.01)
                print(f"클릭 속도 감소: {1/self.click_delay:.1f} CPS")
            elif key == keyboard.Key.esc:  # ESC로 프로그램 종료
                print("프로그램 종료...")
                self.stop()
                return False
        except AttributeError:
            pass
            
    def start(self):
        print("오토클리커 시작!")
        print("사용법:")
        print("- 마우스 좌클릭을 누르고 있으면 자동 연타")
        print("- F1: 클릭 속도 증가")
        print("- F2: 클릭 속도 감소")
        print("- ESC: 프로그램 종료")
        print(f"현재 클릭 속도: {1/self.click_delay:.1f} CPS")
        
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key)
        
        self.mouse_listener.start()
        self.keyboard_listener.start()
        
        self.keyboard_listener.join()
        
    def stop(self):
        self.running = False
        self.clicking = False
        if self.mouse_listener:
            self.mouse_listener.stop()

if __name__ == "__main__":
    clicker = AutoClicker()
    clicker.start()