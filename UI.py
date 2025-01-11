import dearpygui.dearpygui as dpg
import os
import numpy as np
import Simulator as sim

FONT_PATH = os.path.join(os.path.dirname(
    __file__), "Font", "GmarketSansTTFMedium.ttf")
MAIN_TITLE = "운동 분석 시뮬레이터"

MAIN_WINDOW_ID = "main_window"
MAIN_WINDOW_LABEL = "Main Window"
MAIN_THEME_TAG = "custom_theme"
DIM_TAB_TAG = "simultors_of_{}_dimension"
DIM_TAB_LABEL = "{}차원 시뮬레이터"

OPEN_SIMUL_BUTTON_TAG = "open_{}_button"
SIMUL_WINDOW_TAG = "{}_window"
R0_INPUT_TEXT_TAG = "r0_input_{}_of_{}"
V0_INPUT_TEXT_TAG = "v0_input_{}_of_{}"
CONST_INPUT_TEXT_TAG = "{}_input_of_{}"

R0_DEFAULT_VALUE = "0"
V0_DEFAULT_VALUE = "0"
CONST_DEFAULT_VALUE = "0.0"
RUN_SIMUL_BUTTON_LABEL = "Running simulation"


class UIManager:
    """
    사용자 입력을 받는 UI를 담당하는 객체.

    시뮬레이션 매니저에서 시뮬레이션을 등록하면 UI에 등록해준다.
    시뮬레이션을 선택하면 그 시뮬레이션에 알맞는 정보를 입력받고 시뮬레이션 객체로 넘긴다.

    생성 후 start()를 호출하면 UI를 띄움.
    """

    def __init__(self, simulator_manager):
        self.simulator_manager = simulator_manager
        self.main_window_id = MAIN_WINDOW_ID

        dpg.create_context()
        self.setup_fonts()
        self.setup_theme()
        dpg.create_viewport(title=MAIN_TITLE, width=800, height=600)
        self.create_main_window()
        dpg.set_primary_window(MAIN_WINDOW_ID, True)

        simulator_manager.add_event_listener(
            sim.SimulatorManager.AddSimulatorEvent.NAME, self.add_simulator_listener)

    def setup_fonts(self):
        """
        한글 폰트를 등록한다.
        """
        with dpg.font_registry():
            with dpg.font(FONT_PATH, 20, default_font=True) as korean_font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean)
        self.korean_font = korean_font

    def setup_theme(self):
        """
        메인 테마를 설정한다.
        """
        with dpg.theme(tag=MAIN_THEME_TAG):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 10)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 20, 20)

    def get_tab_id(self, dimension):
        """
        Main Window에 있는 Dimension Tab의 ID를 반환한다.
        """
        return DIM_TAB_TAG.format(dimension)

    def add_simulator_listener(self, e):
        """
        SimulatorManager에게서 AddSimulatorEvent가 발생했을 때 호출되는 함수.
        """
        self.add_simulator_button_to_tab(
            e, self.get_tab_id(e.simulator.dimension))

    def add_simulator_button_to_tab(self, e, tab_id):
        """
        tab_id 하위에 클릭하면 시뮬레이터 Window가 열리는 버튼을 추가한다.
        """
        with dpg.group(horizontal=True, parent=tab_id):
            dpg.add_button(label=e.name, tag=OPEN_SIMUL_BUTTON_TAG.format(
                e.name), callback=lambda s, a: self.open_simulator(e.name, e.simulator))

    def open_simulator(self, name, simulator):
        """
        r0, v0, 상수들을 입력받고 시뮬레이션 할 수 있는 Window를 연다.
        """
        window_tag = SIMUL_WINDOW_TAG.format(name)
        if dpg.does_item_exist(window_tag):
            dpg.show_item(window_tag)
        else:
            try:
                with dpg.window(label=name, pos=(100, 100), width=300, height=200, show=True, tag=window_tag, on_close=lambda: dpg.hide_item(window_tag)):
                    with dpg.group(horizontal=True):
                        dpg.add_text("r_0:")
                        for dimension in range(simulator.dimension):
                            dpg.add_input_text(
                                tag=R0_INPUT_TEXT_TAG.format(dimension, name), default_value=R0_DEFAULT_VALUE, width=100)

                    with dpg.group(horizontal=True):
                        dpg.add_text("v_0:")
                        for dimension in range(simulator.dimension):
                            dpg.add_input_text(
                                tag=V0_INPUT_TEXT_TAG.format(dimension, name), default_value=V0_DEFAULT_VALUE, width=100)

                    for const in simulator.consts:
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"{const}:")
                            dpg.add_input_text(
                                tag=CONST_INPUT_TEXT_TAG.format(const, name), default_value=CONST_DEFAULT_VALUE, width=100)

                    dpg.add_button(
                        label=RUN_SIMUL_BUTTON_LABEL, callback=lambda s, a: self.simulate(name, simulator))
            except SystemError as e:
                print(f"Failed to create window: {e}")

    def simulate(self, name, simulator):
        """
        현재 열려있는 Window에서 입력받은 값으로 시뮬레이션을 실행한다.
        """
        try:
            r_0 = np.array([float(dpg.get_value(R0_INPUT_TEXT_TAG.format(
                i, name))) for i in range(simulator.dimension)])
            v_0 = np.array([float(dpg.get_value(V0_INPUT_TEXT_TAG.format(
                i, name))) for i in range(simulator.dimension)])
            consts = {const: float(dpg.get_value(CONST_INPUT_TEXT_TAG.format(
                const, name))) for const in simulator.consts}

            simulator.simulate(r_0, v_0, **consts)
        except ValueError as e:
            print(f"ValueError: {e}")

    def create_main_window(self):
        """
        Main Window를 생성한다.
        """
        with dpg.window(label=MAIN_WINDOW_LABEL, width=800, height=600, pos=(0, 0), no_title_bar=True, tag=self.main_window_id):
            dpg.bind_item_theme(MAIN_WINDOW_ID, MAIN_THEME_TAG)
            with dpg.tab_bar():
                with dpg.tab(label=DIM_TAB_LABEL.format(1), tag=DIM_TAB_TAG.format(1)):
                    dpg.add_spacer(height=4)
                with dpg.tab(label=DIM_TAB_LABEL.format(2), tag=DIM_TAB_TAG.format(2)):
                    dpg.add_spacer(height=4)
                with dpg.tab(label=DIM_TAB_LABEL.format(3), tag=DIM_TAB_TAG.format(3)):
                    dpg.add_spacer(height=4)

    def start(self):
        """
        UI를 사용자에게 띄운다.
        """
        dpg.bind_font(self.korean_font)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
