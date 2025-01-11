from abc import ABC, abstractmethod
from matplotlib.animation import FuncAnimation
import EventEmitter as ee
import matplotlib.pyplot as plt
import numpy as np
import dearpygui.dearpygui as dpg


DEFAULT_POSITION_AXES_LIMIT = (-2 * np.pi, 2 * np.pi)
DEFAULT_VELOCITY_AXES_LIMIT = (-10, 10)
DEFAULT_TIME_AXES_LIMIT = (0, 4 * np.pi)


def add_data(line, new_x, new_y):
    """
    Line2D 객체의 Data Set에 새로운 데이터를 추가한다.
    """
    old_x, old_y = line.get_data()

    new_x = np.array(new_x).flatten()
    new_y = np.array(new_y).flatten()

    updated_x = np.concatenate((old_x, new_x))
    updated_y = np.concatenate((old_y, new_y))

    line.set_data(updated_x, updated_y)
    return line


class SimulatorManager(ee.EventEmitter):
    """
    시뮬레이터 객체들을 관리하는 클래스.

    시뮬레이터를 등록하면 List에 저장하고, AddSimulator Event를 발생시킨다.
    """
    class AddSimulatorEvent:
        NAME = "Add Simulator"

        def __init__(self, name, simulator):
            self.name = name
            self.simulator = simulator

    def __init__(self):
        self.simulators = {}
        self.__event_listeners = []

    def add_simulator(self, name, simulator):
        """
        시뮬레이터 등록
        """
        self.simulators[name] = simulator
        self.notify_event_listeners(
            SimulatorManager.AddSimulatorEvent.NAME, SimulatorManager.AddSimulatorEvent(name, simulator))

    def add_event_listener(self, event_name, event_listener):
        """
        이벤트 리스너 등록
        """
        self.__event_listeners.append(
            {"event_listener": event_listener, "event_name": event_name})

    def notify_event_listeners(self, event_name, event_object):
        """
        이벤트 리스너에게 event_name 이벤트 발생을 알림.
        각 이벤트 리스너에게 event_object를 전달함.
        """
        for listener in self.__event_listeners:
            if listener["event_name"] == event_name:
                listener["event_listener"](event_object)


class Simulator(ABC):
    """
    시뮬레이터 객체.

    r(현재 위치)와 v(현재 속도)는 항상 numpy array로 관리되어야 한다.
    """

    def __init__(self):
        self.r = np.array([0])
        self.v = np.array([0])
        self.t_max = 10000
        self.dt = 0.01
        self.consts = []

    @abstractmethod
    def get_dv_dt(self):
        pass

    @abstractmethod
    def simulate(self, r_0: np.ndarray, v_0: np.ndarray):
        """
        matplotlib plotting과 matplotlib animation을 이용하여
        입자의 운동을 실시간으로 시뮬레이션한다.

        r_0: numpy array, 입자의 초기 위치
        v_0: numpy array, 입자의 초기 속도

        Figure(캔버스)를 생성하고, 위상 공간이 그려질 Axes와 실제 공간이 그려질 Axes를 생성한다.

        물체의 운동이 1차원에 국한될 경우
        - Phase Axes in x-v Line2D
        - Real Axes in x-t Scatter

        물체의 운동이 2차원에 국한될 경우
        - Phase Axes in x-v_x Line2D, y-v_y Line2D
        - Real Axes in x-y Scatter

        물체의 운동이 3차원에 국한될 경우
        - Phase Axes in x-v_x Line2D, y-v_y Line2D, z-v_z Line2D
        - Real Axes in x-y-z Scatter

        초기화 함수와 r += v * dt, v += dv/dt * dt를 이용하여 업데이트 함수를 정의한다.
        업데이트 함수에서, 입자나 Line이 그래프의 범위를 넘어가면 그래프의 범위를 재설정하는 로직이 있어야 한다.

        이후 FuncAnimation을 이용하여 시뮬레이션을 실행한다.
        """


class SimulatorInOneDim(Simulator):
    def __init__(self):
        super().__init__()
        self.dimension = 1

    def get_x(self):
        return self.r[0]

    def get_v(self):
        return self.v[0]

    def simulate(self, r_0: np.ndarray, v_0: np.ndarray):
        self.r = r_0
        self.v = v_0

        fig, ax = plt.subplots(2, figsize=(10, 10))
        phase_axes, real_axes = ax[0], ax[1]

        line = phase_axes.plot(
            [self.get_x()], [self.get_v()], label='x-v Path')[0]
        particle = real_axes.scatter(
            [0], [self.get_x()], label='x-t Path')

        def init():
            return line, particle

        def update(frame):
            dv = self.get_dv_dt() * self.dt
            self.v = self.v + dv
            self.r = self.r + self.v * self.dt

            add_data(line, self.r, self.v)
            particle.set_offsets([[frame * self.dt, self.get_x()]])

            # 그래프 범위 동적 조정 및 축 라벨 업데이트
            xlim = phase_axes.get_xlim()
            ylim = phase_axes.get_ylim()
            if self.get_x() > 0.85 * xlim[1] or self.get_x() < 0.85 * xlim[0]:
                phase_axes.set_xlim(
                    min(xlim[0], self.get_x() * 1.15), max(xlim[1], self.get_x() * 1.15))
                phase_axes.figure.canvas.draw()

            if self.get_v() > 0.85 * ylim[1] or self.get_v() < 0.85 * ylim[0]:
                phase_axes.set_ylim(
                    min(ylim[0], self.get_v() * 1.15), max(ylim[1], self.get_v() * 1.15))
                phase_axes.figure.canvas.draw()

            xlim = real_axes.get_xlim()
            ylim = real_axes.get_ylim()
            if frame * self.dt > 0.85 * xlim[1] or frame * self.dt < 0.85 * xlim[0]:
                real_axes.set_xlim(
                    min(xlim[0], frame * self.dt * 1.15), max(xlim[1], frame * self.dt * 1.15))
                real_axes.figure.canvas.draw()

            if self.get_x() > 0.85 * ylim[1] or self.get_x() < 0.85 * ylim[0]:
                real_axes.set_ylim(
                    min(ylim[0], self.get_x() * 1.15), max(ylim[1], self.get_x() * 1.15))
                real_axes.figure.canvas.draw()

            return line, particle

        ani = FuncAnimation(fig, update, frames=int(self.t_max / self.dt),
                            interval=self.dt * 1000, repeat=False, init_func=init, blit=True)

        phase_axes.set_xlim(*DEFAULT_POSITION_AXES_LIMIT)
        phase_axes.set_ylim(*DEFAULT_VELOCITY_AXES_LIMIT)
        phase_axes.set_xlabel('x (m)')
        phase_axes.set_ylabel('v (m/s)')
        phase_axes.set_title('Phase Space')
        phase_axes.legend()

        real_axes.set_xlim(*DEFAULT_TIME_AXES_LIMIT)
        real_axes.set_ylim(*DEFAULT_POSITION_AXES_LIMIT)
        real_axes.set_xlabel('t (s)')
        real_axes.set_ylabel('x (m)')
        real_axes.set_title('Real Space')
        real_axes.legend()

        plt.tight_layout()
        plt.show()


class SimulatorInTwoDim(Simulator):
    def __init__(self):
        super().__init__()
        self.dimension = 2

    def get_x(self):
        return self.r[0]

    def get_y(self):
        return self.r[1]

    def get_vx(self):
        return self.v[0]

    def get_vy(self):
        return self.v[1]

    def simulate(self, r_0: np.ndarray, v_0: np.ndarray):
        self.r = r_0
        self.v = v_0

        fig, ax = plt.subplots(2, figsize=(10, 10))
        phase_axes, real_axes = ax[0], ax[1]

        x_vx_line = phase_axes.plot(
            [self.get_x()], [self.get_vx()], label='x-v_x Path')[0]
        y_vy_line = phase_axes.plot(
            [self.get_y()], [self.get_vy()], label='y-v_y Path')[0]
        particle = real_axes.scatter(
            [self.get_x()], [self.get_y()], label='x-y Path')

        def init():
            return x_vx_line, y_vy_line, particle

        def update(frame):
            dv_dt = self.get_dv_dt()
            dv = dv_dt * self.dt
            self.v = self.v + dv
            self.r = self.r + self.v * self.dt

            add_data(x_vx_line, [self.get_x()], [self.get_vx()])
            add_data(y_vy_line, [self.get_y()], [self.get_vy()])
            particle.set_offsets(
                np.array([[self.get_x(), self.get_y()]]))

            # 그래프 범위 동적 조정 및 축 라벨 업데이트
            xlim = phase_axes.get_xlim()
            ylim = phase_axes.get_ylim()
            if self.get_x() > 0.85 * xlim[1] or self.get_x() < 0.85 * xlim[0]:
                phase_axes.set_xlim(
                    min(xlim[0], self.get_x() * 1.15), max(xlim[1], self.get_x() * 1.15))
                phase_axes.figure.canvas.draw()

            if self.get_vx() > 0.85 * ylim[1] or self.get_vx() < 0.85 * ylim[0]:
                phase_axes.set_ylim(
                    min(ylim[0], self.get_vx() * 1.15), max(ylim[1], self.get_vx() * 1.15))
                phase_axes.figure.canvas.draw()

            xlim = real_axes.get_xlim()
            ylim = real_axes.get_ylim()
            if self.get_x() > 0.85 * xlim[1] or self.get_x() < 0.85 * xlim[0]:
                real_axes.set_xlim(
                    min(xlim[0], self.get_x() * 1.15), max(xlim[1], self.get_x() * 1.15))
                real_axes.figure.canvas.draw()

            if self.get_y() > 0.85 * ylim[1] or self.get_y() < 0.85 * ylim[0]:
                real_axes.set_ylim(
                    min(ylim[0], self.get_y() * 1.15), max(ylim[1], self.get_y() * 1.15))
                real_axes.figure.canvas.draw()

            return x_vx_line, y_vy_line, particle

        ani = FuncAnimation(fig, update, frames=int(self.t_max / self.dt),
                            interval=self.dt * 1000, repeat=False, init_func=init, blit=True)

        phase_axes.set_xlim(*DEFAULT_POSITION_AXES_LIMIT)
        phase_axes.set_ylim(*DEFAULT_VELOCITY_AXES_LIMIT)
        phase_axes.set_xlabel('r (m)')
        phase_axes.set_ylabel('v (m/s)')
        phase_axes.set_title('Phase Space')
        phase_axes.legend()

        real_axes.set_xlim(*DEFAULT_POSITION_AXES_LIMIT)
        real_axes.set_ylim(*DEFAULT_POSITION_AXES_LIMIT)
        real_axes.set_xlabel('x (m)')
        real_axes.set_ylabel('y (m)')
        real_axes.set_title('Real Space')
        real_axes.legend()

        plt.tight_layout()
        plt.show()


class SimulatorInThreeDim(Simulator):
    def __init__(self):
        super().__init__()
        self.dimension = 3

    def get_x(self):
        return self.r[0]

    def get_y(self):
        return self.r[1]

    def get_z(self):
        return self.r[2]

    def get_vx(self):
        return self.v[0]

    def get_vy(self):
        return self.v[1]

    def get_vz(self):
        return self.v[2]

    def simulate(self, r_0: np.ndarray, v_0: np.ndarray):
        self.r = r_0
        self.v = v_0

        fig = plt.figure(figsize=(10, 10))
        phase_axes = fig.add_subplot(121, projection='3d')
        real_axes = fig.add_subplot(122, projection='3d')

        x_vx_line = phase_axes.plot(
            [self.get_x()], [self.get_vx()], label='x-v_x Path')[0]
        y_vy_line = phase_axes.plot(
            [self.get_y()], [self.get_vy()], label='y-v_y Path')[0]
        z_vz_line = phase_axes.plot(
            [self.get_z()], [self.get_vz()], label='z-v_z Path')[0]
        particle = real_axes.scatter([self.get_x()], [self.get_y()], [
            self.get_z()], label='x-y-z Path')

        def init():
            return x_vx_line, y_vy_line, z_vz_line, particle

        def update(frame):
            dv_dt = self.get_dv_dt()
            dv = dv_dt * self.dt
            self.v = self.v + dv
            self.r = self.r + self.v * self.dt

            add_data(x_vx_line, [self.get_x()], [self.get_vx()])
            add_data(y_vy_line, [self.get_y()], [self.get_vy()])
            add_data(z_vz_line, [self.get_z()], [self.get_vz()])
            particle._offsets3d = (
                self.get_x(), self.get_y(), self.get_z())

            return x_vx_line, y_vy_line, z_vz_line, particle

        ani = FuncAnimation(fig, update, frames=int(self.t_max / self.dt),
                            interval=self.dt * 1000, repeat=False, init_func=init, blit=True)

        phase_axes.set_xlim(*DEFAULT_POSITION_AXES_LIMIT)
        phase_axes.set_ylim(*DEFAULT_VELOCITY_AXES_LIMIT)
        phase_axes.set_xlabel('r (m)')
        phase_axes.set_ylabel('v (m/s)')
        phase_axes.set_title('Phase Space')
        phase_axes.legend()

        real_axes.set_xlim(*DEFAULT_POSITION_AXES_LIMIT)
        real_axes.set_ylim(*DEFAULT_POSITION_AXES_LIMIT)
        real_axes.set_zlim(*DEFAULT_POSITION_AXES_LIMIT)
        real_axes.set_xlabel('x (m)')
        real_axes.set_ylabel('y (m)')
        real_axes.set_zlabel('z (m)')
        real_axes.set_title('Real Space')
        real_axes.legend()

        plt.tight_layout()
        plt.show()


"""

시뮬레이션 템플릿은 다음과 같이 생성할 수 있다.

1. SimulatorInOneDim or SimulatorInTwoDim or SimulatorInThreeDim을 상속받는다.
2. __init__에서 super().__init__()를 호출 후, 필요한 상수들을 선언한다.
3. a=F/m을 이용하여 get_dv_dt()를 오버라이딩한다.
4. simulate 함수의 인자로 r_0, v_0과 상수값들을 받아서 상수값들은 self에 저장하고, super().simulate(r_0, v_0)을 실행한다.

변수가 x, y, z가 아닌 theta와 같이 다른 변수를 사용하고 싶은 경우 r과 v에 theta, w를 넣어 사용한다. 
다만 theta와 x, y, z 변수와의 관계를 찾아서 get_x, get_y, get_z, get_vx, get_vy, get_vz를 오버라이딩해야 한다.

"""


class SimpleHarmonicSimulator(SimulatorInOneDim):
    def __init__(self):
        super().__init__()
        self.consts = ['k', 'm']
        self.k = 0
        self.m = 0

    def simulate(self, x_0, v_0, k, m):
        self.k = k
        self.m = m
        super().simulate(x_0, v_0)

    def get_dv_dt(self):
        return -self.k * self.r / self.m


class DampedHarmonicSimulator(SimulatorInOneDim):
    def __init__(self):
        super().__init__()
        self.consts = ['k', 'm', 'c']
        self.k = 0
        self.m = 0
        self.c = 0

    def simulate(self, x_0, v_0, k, m, c):
        self.k = k
        self.m = m
        self.c = c
        super().simulate(x_0, v_0)

    def get_dv_dt(self):
        return -self.k * self.r / self.m - self.c * self.v / self.m


class FreeFallSimulator(SimulatorInOneDim):
    def __init__(self):
        super().__init__()
        self.consts = ['g']
        self.g = 0

    def simulate(self, x_0, v_0, g):
        self.g = g
        super().simulate(x_0, v_0)

    def get_dv_dt(self):
        return -self.g


class PendulumSimulator(SimulatorInTwoDim):
    def __init__(self):
        super().__init__()
        self.consts = ['g', 'l', 'c', 'm']
        self.g = 0
        self.l = 0
        self.c = 0
        self.m = 0

    def simulate(self, theta_0, w_0, g, l, c, m):
        self.g = g
        self.l = l
        self.c = c
        self.m = m
        super().simulate(theta_0, w_0)

    def get_theta(self):
        return self.r[0]

    def get_w(self):
        return self.v[0]

    def get_x(self):
        return self.l * np.sin(self.get_theta())

    def get_y(self):
        return -self.l * np.cos(self.get_theta())

    def get_vx(self):
        return self.l * np.cos(self.get_theta()) * self.get_w()

    def get_vy(self):
        return self.l * np.sin(self.get_theta()) * self.get_w()

    def get_dv_dt(self):
        return -self.g / self.l * np.sin(self.get_theta())


class DampedPendulumSimulator(SimulatorInTwoDim):
    def __init__(self):
        super().__init__()
        self.consts = ['g', 'l', 'c', 'm']
        self.g = 0
        self.l = 0
        self.c = 0
        self.m = 0

    def simulate(self, theta_0, w_0, g, l, c, m):
        self.g = g
        self.l = l
        self.c = c
        self.m = m
        super().simulate(theta_0, w_0)

    def get_theta(self):
        return self.r[0]

    def get_w(self):
        return self.v[0]

    def get_x(self):
        return self.l * np.sin(self.get_theta())

    def get_y(self):
        return -self.l * np.cos(self.get_theta())

    def get_vx(self):
        return self.l * np.cos(self.get_theta()) * self.get_w()

    def get_vy(self):
        return self.l * np.sin(self.get_theta()) * self.get_w()

    def get_dv_dt(self):
        return -self.g / self.l * np.sin(self.get_theta()) - self.c * self.get_w() / self.m


class CircularMotionSimulator(SimulatorInTwoDim):
    def __init__(self):
        super().__init__()
        self.consts = ['radius']
        self.radius = 0  # 반지름 (m)

    def simulate(self, theta_0, w_0, radius):
        self.radius = radius
        super().simulate(theta_0, w_0)

    def get_theta(self):
        return self.r[0]

    def get_w(self):
        return self.v[0]

    def get_x(self):
        # x = r * cos(theta)
        return self.radius * np.cos(self.get_theta())

    def get_y(self):
        # y = r * sin(theta)
        return self.radius * np.sin(self.get_theta())

    def get_vx(self):
        # vx = -r * omega * sin(theta)
        return -self.radius * self.get_w() * np.sin(self.get_theta())

    def get_vy(self):
        # vy = r * omega * cos(theta)
        return self.radius * self.get_w() * np.cos(self.get_theta())

    def get_dv_dt(self):
        # 원운동의 경우 가속도는 중심을 향하는 구심 가속도입니다.
        dvx_dt = -self.get_w()**2 * self.get_x()
        dvy_dt = -self.get_w()**2 * self.get_y()
        return np.array([dvx_dt, dvy_dt])


class ProjectileSimulator(SimulatorInTwoDim):
    def __init__(self):
        super().__init__()
        self.consts = ['g']
        self.g = 0

    def simulate(self, r_0, v_0, g):
        self.g = g
        super().simulate(r_0, v_0)

    def get_dv_dt(self):
        return np.array([0, -self.g])


class DampedProjectileSimulator(SimulatorInTwoDim):
    def __init__(self):
        super().__init__()
        self.consts = ['g', 'b', 'm']
        self.g = 9.81  # 중력 가속도 (m/s^2)
        self.b = 0  # 감쇠 계수 (kg/s)
        self.m = 0  # 질량 (kg)

    def simulate(self, r_0, v_0, g, b, m):
        self.g = g
        self.b = b
        self.m = m
        super().simulate(r_0, v_0)

    def get_dv_dt(self):
        dvx_dt = -self.b * self.get_vx() / self.m
        dvy_dt = -self.g - self.b * self.get_vy() / self.m
        return np.array([dvx_dt, dvy_dt])


class SimpleHarmonicSimulator3D(SimulatorInThreeDim):
    def __init__(self):
        super().__init__()
        self.consts = ['k', 'm']
        self.k = 0
        self.m = 0

    def simulate(self, x_0, v_0, k, m):
        self.k = k
        self.m = m
        super().simulate(x_0, v_0)

    def get_dv_dt(self):
        return -self.k * self.r / self.m
