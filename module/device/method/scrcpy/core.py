import socket
import struct
import threading
import time
import typing as t
from time import sleep

import numpy as np
from adbutils import _AdbStreamConnection, AdbError, Network

from module.base.decorator import cached_property
from module.base.timer import Timer
from module.device.connection import Connection
from module.device.method.scrcpy.control import ControlSender
from module.device.method.scrcpy.options import ScrcpyOptions
from module.exception import RequestHumanTakeover
from module.logger import logger


class ScrcpyError(Exception):
    pass


class ScrcpyCore(Connection):
    """
    Scrcpy: https://github.com/Genymobile/scrcpy
    Module from https://github.com/leng-yue/py-scrcpy-client
    """

    _scrcpy_last_frame: t.Optional[np.ndarray] = None
    _scrcpy_last_frame_time: float = 0.

    _scrcpy_alive = False
    _scrcpy_initialized = False
    _scrcpy_server_stream: t.Optional[_AdbStreamConnection] = None
    _scrcpy_video_socket: t.Optional[socket.socket] = None
    _scrcpy_control_socket: t.Optional[socket.socket] = None
    _scrcpy_control_socket_lock = threading.Lock()

    _scrcpy_stream_loop_thread = None
    _scrcpy_resolution: t.Tuple[int, int] = (1280, 720)

    @cached_property
    def _scrcpy_control(self) -> ControlSender:
        return ControlSender(self)

    @property
    def _scrcpy_jar_local(self) -> str:
        # Fall back to the bundled jar in case config_manual.py was not updated.
        return self.config.SCRCPY_FILEPATH_LOCAL or './bin/scrcpy/scrcpy-server-v1.20.jar'

    @property
    def _scrcpy_jar_remote(self) -> str:
        return self.config.SCRCPY_FILEPATH_REMOTE or '/data/local/tmp/scrcpy-server.jar'

    def scrcpy_init(self):
        self._scrcpy_initialized = False
        if not self.config.SCRCPY_FILEPATH_LOCAL or not self.config.SCRCPY_FILEPATH_REMOTE:
            logger.warning(
                'SCRCPY_FILEPATH_LOCAL or SCRCPY_FILEPATH_REMOTE is not set, '
                'using the bundled scrcpy-server-v1.20.jar'
            )
        self._scrcpy_server_stop()
        self._scrcpy_server_kill_stale()

        logger.hr('Scrcpy init')
        logger.info(f'pushing {self._scrcpy_jar_local}')
        self.adb_push(self._scrcpy_jar_local, self._scrcpy_jar_remote)
        # Host archives may preserve a restrictive source mode on this file.
        # app_process runs as shell and cannot load a 0000-mode jar.
        self.adb_shell(['chmod', '644', self._scrcpy_jar_remote])

        self._scrcpy_alive = False
        self._scrcpy_initialized = True
        self.scrcpy_ensure_running()

    def _scrcpy_server_kill_stale(self):
        """
        Kill leftover scrcpy server processes on the device.

        If a previous scrcpy session crashed without a clean shutdown, the old
        server process keeps the abstract socket bound, so the new server aborts
        with "Address already in use". Only kill processes running scrcpy.Server.
        """
        logger.info('Kill stale scrcpy server on device')
        try:
            self.adb_shell(
                'ps -A -o PID,NAME,ARGS | awk '
                '\'$2=="app_process" || $2=="app_process32" || $2=="app_process64" '
                '{if ($0 ~ /com.genymobile.scrcpy.Server/) print $1}\' '
                '| xargs -r kill -9 2>/dev/null; true'
            )
        except Exception as e:
            logger.warning(f'Unable to kill stale scrcpy server: {e}')

    def scrcpy_ensure_running(self):
        if not self._scrcpy_alive:
            if not self._scrcpy_initialized:
                self.scrcpy_init()
                return
            with self._scrcpy_control_socket_lock:
                self._scrcpy_server_start()

    def _scrcpy_server_start(self):
        """
        Connect to scrcpy server, there will be two sockets, video and control socket.

        Raises:
            ScrcpyError:
        """
        logger.hr('Scrcpy server start')
        # Never return a frame from the previous server session.
        self._scrcpy_last_frame = None
        self._scrcpy_last_frame_time = 0.
        commands = ScrcpyOptions.command_v120(jar_path=self._scrcpy_jar_remote)
        self._scrcpy_server_stream: _AdbStreamConnection = self.adb.shell(
            commands,
            stream=True,
        )

        logger.info('Create server stream')
        # Do not block on the server stream. On some emulators (e.g. Android 12)
        # the server logs go to logcat, and stdout stays empty (buffered) while
        # the server is running, so a blocking read would hang forever.
        ret = self._scrcpy_receive_from_server_stream()
        if ret:
            # b'Aborted \r\n'
            # Probably because file not exists
            if b'Aborted' in ret:
                raise ScrcpyError('Aborted')
            if ret.startswith(b'[server] E'):
                # [server] ERROR: ...
                ret += self._scrcpy_receive_from_server_stream() or b''
                logger.error(ret)
                # java.lang.IllegalArgumentException: The server version (1.25) does not match the client (...)
                if b'does not match the client' in ret:
                    raise ScrcpyError('Server version does not match the client')
                else:
                    raise ScrcpyError('Unknown scrcpy error')
            else:
                # [server] INFO: Device: ...
                ret += self._scrcpy_receive_from_server_stream() or b''
                logger.info(ret)

        logger.info('Create video socket')
        timeout = Timer(3).start()
        while 1:
            if timeout.reached():
                raise ScrcpyError('Connect scrcpy-server timeout')

            try:
                self._scrcpy_video_socket = self.adb.create_connection(
                    Network.LOCAL_ABSTRACT, "scrcpy"
                )
                break
            except AdbError:
                sleep(0.1)
        dummy_byte = self._scrcpy_video_socket.recv(1)
        if not len(dummy_byte) or dummy_byte != b"\x00":
            raise ScrcpyError('Did not receive Dummy Byte from video stream')

        logger.info('Create control socket')
        self._scrcpy_control_socket = self.adb.create_connection(
            Network.LOCAL_ABSTRACT, "scrcpy"
        )

        logger.info('Fetch device info')
        device_name = self._scrcpy_video_socket.recv(64).decode("utf-8").rstrip("\x00")
        if len(device_name):
            logger.attr('Scrcpy Device', device_name)
        else:
            raise ScrcpyError('Did not receive Device Name')
        ret = self._scrcpy_video_socket.recv(4)
        self._scrcpy_resolution = struct.unpack(">HH", ret)
        logger.attr('Scrcpy Resolution', self._scrcpy_resolution)

        self._scrcpy_video_socket.setblocking(False)
        self._scrcpy_alive = True

        logger.info('Start video stream loop thread')
        self._scrcpy_stream_loop_thread = threading.Thread(
            target=self._scrcpy_stream_loop, daemon=True
        )
        self._scrcpy_stream_loop_thread.start()
        while 1:
            if self._scrcpy_stream_loop_thread is not None and self._scrcpy_stream_loop_thread.is_alive():
                break
            self.sleep(0.001)

        logger.info('Scrcpy server is up')

    def _scrcpy_server_stop(self):
        """
        Stop listening (both threaded and blocked)
        """
        logger.hr('Scrcpy server stop')
        # err = self._scrcpy_receive_from_server_stream()
        # if err:
        #     logger.error(err)

        self._scrcpy_alive = False
        if self._scrcpy_server_stream is not None:
            try:
                self._scrcpy_server_stream.close()
            except Exception:
                pass

        if self._scrcpy_control_socket is not None:
            try:
                self._scrcpy_control_socket.close()
            except Exception:
                pass

        if self._scrcpy_video_socket is not None:
            try:
                self._scrcpy_video_socket.close()
            except Exception:
                pass
        self._scrcpy_server_stream = None
        self._scrcpy_control_socket = None
        self._scrcpy_video_socket = None

        logger.info('Scrcpy server stopped')

    def _scrcpy_receive_from_server_stream(self, timeout: float = 1.0):
        if self._scrcpy_server_stream is not None:
            try:
                conn = self._scrcpy_server_stream.conn
                try:
                    old_timeout = conn.gettimeout()
                except Exception:
                    old_timeout = None
                conn.settimeout(timeout)
                try:
                    return conn.recv(4096)
                finally:
                    try:
                        conn.settimeout(old_timeout)
                    except Exception:
                        pass
            except (socket.timeout, TimeoutError):
                pass
            except Exception:
                pass

    def _scrcpy_stream_loop(self) -> None:
        """
        Core loop for video parsing
        """
        try:
            from av.codec import CodecContext
            from av.error import InvalidDataError
        except ImportError as e:
            logger.error(e)
            logger.error('You must have `av` installed to use scrcpy screenshot, please update dependencies')
            raise RequestHumanTakeover

        codec = CodecContext.create("h264", "r")
        while self._scrcpy_alive:
            try:
                raw_h264 = self._scrcpy_video_socket.recv(0x10000)
                if raw_h264 == b"":
                    raise ScrcpyError("Video stream is disconnected")
                packets = codec.parse(raw_h264)
                for packet in packets:
                    frames = codec.decode(packet)
                    for frame in frames:
                        # logger.info('frame received')
                        frame = frame.to_ndarray(format="rgb24")
                        self._scrcpy_last_frame = frame
                        self._scrcpy_last_frame_time = time.time()
                        self._scrcpy_resolution = (frame.shape[1], frame.shape[0])
            except (BlockingIOError, InvalidDataError):
                # only return nonempty frames, may block cv2 render thread
                time.sleep(0.001)
            except (ConnectionError, OSError) as e:  # Socket Closed
                if self._scrcpy_alive:
                    logger.error(f'_scrcpy_stream_loop_thread: {repr(e)}')
                    raise

        raise ScrcpyError('_scrcpy_stream_loop stopped')
