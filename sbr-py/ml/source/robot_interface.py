import threading
from enum import Enum
import logging
import numpy as np

# connectivity setup
from util.connectivity import Connectivity

uart_port = 'COM3'  # in case of UART connectivity
uart_speed = 115200  # serial port speed
wifi_local_ip = '192.168.4.1'  # host (this) IP
wifi_robot_ip = '192.158.4.2'  # remote (robot) IP
wifi_local_port = '1234'  # host (this) port
wifi_robot_port = '1235'  # remote (robot) port


class ConnectionType(Enum):
    UART = 1
    WIFI = 2


VALUES = {0: -210, 1: -32, 2: 0, 3: 32, 4: 210}
VALUES_RANGE = 5


class RobotInterface:
    def __init__(self):
        parameters = {'port': uart_port, 'speed': uart_speed, 'timeout': 0.01}
        connectivity = 'UART'
        self._con = Connectivity(connectivity, parameters)
        self._con.write({'type': 'MPUrate', 'rate': 100000})

        self.RAW_ZERO = -1.45
        self.RAW_MIN_RANGE = -12
        self.RAW_MAX_RANGE = 8

        self._last_value = None
        self._stop_flag = False
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._update)
        self._thread.start()

    def _update(self):
        while not self._stop_flag:
            msg = self._con.read()
            if msg['type'] == 'MPUdata':
                # print("------ {}".format(msg['acc_y']))
                logging.debug('acc: {: >5.2f} {: >5.2f} {: >5.2f}, gyro:  {: >5.2f} {: >5.2f} {: >5.2f}'
                              .format(msg['acc_x'], msg['acc_y'], msg['acc_z'], msg['gyro_x'], msg['gyro_y'],
                                      msg['gyro_z']))
                with self._lock:
                    self._last_value = (msg['acc_y'], msg['gyro_x'])

    def getState(self):
        with self._lock:
            return self._last_value

    def setState(self, vel: np.ndarray):
        if vel[0] not in VALUES:
            print('Bad value')
            return
        self._con.write({'type': 'SetMotors', 'left': VALUES[vel[0]], 'right': -VALUES[vel[0]]})

    def stop(self):
        self._stop_flag = True
