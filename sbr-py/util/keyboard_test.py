# -*- coding: utf-8 -*-
#
# Description:  communicate with robot (Arduino) read and send command
# Author:       Jaroslaw Bulat (kwant@agh.edu.pl, kwanty@gmail.com)
# Created:      29.01.2021
# License:      GPLv3
# File:         keyboard_test.py
# TODO:
#   -

import sys
from util.connectivity import Connectivity

# connectivity setup
uart_port = 'COM3'  # in case of UART connectivity
uart_speed = 115200  # serial port speed
wifi_local_ip = '192.168.4.1'  # host (this) IP
wifi_robot_ip = '192.158.4.2'  # remote (robot) IP
wifi_local_port = '1234'  # host (this) port
wifi_robot_port = '1235'  # remote (robot) port


def main_loop(connectivity):
    """
    Main loop sending/receiving data.
    When obtain correct frame, print acc/gyro parameters
    :param connectivity: Connectivity object
    """

    # TODO: primitive, inefficient pooling-type bidirectional communication
    left_wheel_speed = 0
    right_wheel_speed = 0
    while True:
        # print('.')
        # print MPU readings
        msg = connectivity.read()
        if msg['type'] is None:  # ignore None msg
            # print("None")
            pass
        elif msg['type'] == 'MPUdata':
            print('acc: {: >5.2f} {: >5.2f} {: >5.2f}, gyro:  {: >5.2f} {: >5.2f} {: >5.2f}'
                  .format(msg['acc_x'], msg['acc_y'], msg['acc_z'], msg['gyro_x'], msg['gyro_y'], msg['gyro_z']))
        else:
            print('Unsupported message from robot, type: {}'.format(msg['type']))


if __name__ == "__main__":
    # script argument could be 'WIFI', 'BT', 'UART' (default with ttyUSB0)
    try:
        connectivity = sys.argv[1]
    except IndexError:
        connectivity = 'UART'

    if connectivity == 'WIFI':
        parameters = {'local_ip': wifi_local_ip, 'robot_ip': wifi_robot_ip, 'local_port': wifi_robot_port,
                      'robot_port': wifi_robot_port}
    elif connectivity == 'BT':
        parameters = {'TBD'}
    elif connectivity == 'UART':
        parameters = {'port': uart_port, 'speed': uart_speed, 'timeout': 0.01}
    else:
        assert False, 'unsupported connectivity method: {}'.format(connectivity)

    con = Connectivity(connectivity, parameters)
    con.write({'type': 'MPUrate', 'rate': 100000})
    con.write({'type': 'SetMotors', 'left': 255, 'right': -255})

    main_loop(con)
