from hid import Device


class TransportUSBHID:
    """USB HID transport through cython-hidapi"""

    def __init__(self, vid, pid):
        # Create an instance of Device class from hid lib
        self._hid_device = Device(vid=vid, pid=pid)

    def write(self, command):
        command_length = len(command) + 1  # (+1 for CRC8 byte)
        # init hid command to report id (0) plus 64 0xFF
        hid_buf = [0x00] + [0xFF] * 64
        # Add header specifying command length
        command = [command_length] + command
        # Add a CRC8 byte at the end
        command = command + [sum(command) % 0x100]
        # Insert the fully fledged command into the data sequence
        hid_buf[1:command_length] = command

        # Send it
        try:
            self._hid_device.write(
                hid_buf
            )  # TODO: Check if there is a response after write
        except:
            raise RuntimeError(f"HID send failed")

        # Read back the response
        resp = self._hid_device.read(64)

        # First byte is the response length
        resp = resp[1:]
        return resp
