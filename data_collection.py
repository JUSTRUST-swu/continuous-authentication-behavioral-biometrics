from pynput import keyboard, mouse
import time, json

OUT = "events.jsonl"

def write_event(e):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

def now():
    return time.perf_counter()

def on_press(key):
    write_event({
        "type": "keydown",
        "key": str(key),
        "t": now()
    })

def on_release(key):
    write_event({
        "type": "keyup",
        "key": str(key),
        "t": now()
    })

def on_move(x, y):
    write_event({
        "type": "mousemove",
        "x": x,
        "y": y,
        "t": now()
    })

def on_click(x, y, button, pressed):
    write_event({
        "type": "mousedown" if pressed else "mouseup",
        "button": str(button),
        "x": x,
        "y": y,
        "t": now()
    })

keyboard_listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)

mouse_listener = mouse.Listener(
    on_move=on_move,
    on_click=on_click
)

keyboard_listener.start()
mouse_listener.start()

keyboard_listener.join()
mouse_listener.join()