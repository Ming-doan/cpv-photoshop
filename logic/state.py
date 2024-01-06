from typing import Callable


class State:
    def __init__(self, init_value):
        self.func: Callable = None
        self.state = init_value

    def register_func(self, func: Callable):
        self.func = func

    def config(self, **kwargs):
        self.func(**kwargs)

    def set(self, callback: Callable):
        new_state = callback(self.state)
        self.state = new_state


class TkState:
    def __init__(self):
        self.states: dict[State] = {}

    def register_state(self, state_name: str, init_value=None) -> State:
        self.states[state_name] = State(init_value)
        return self.states[state_name]

    def get_state(self, state_name: str) -> State:
        try:
            return self.states[state_name]
        except:
            raise KeyError("State not found")


# App State

state = TkState()

canvas_color = state.register_state('canvas_color')

object_list = state.register_state('object_list', [])

selected_object = state.register_state('select_object')

selected_object_text = state.register_state('selected_object_text')

selected_object_edit_btn = state.register_state(
    'selected_object_edit_btn', True)
