from unittest import mock, TestCase
from unittest.mock import patch, MagicMock
from util.connectivity import Connectivity


class TestRobotInterface(TestCase):
    def setUp(self) -> None:
        pass

    @patch('serial.Serial', autospec=True)
    @patch.object(Connectivity, 'read', MagicMock(return_value=8.1))
    def test_01(self, mock_serial):
        conn = Connectivity('UART', {'port': 'TEST', 'speed': 0, 'timeout': 0.01})
        res = conn.read()
        self.assertEqual(res, 8.1)

    def tearDown(self) -> None:
        pass
