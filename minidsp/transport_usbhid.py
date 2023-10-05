import hid


class TransportUSBHID:
    """USB HID transport through cython-hidapi"""

    def __init__(self, vid, pid):
        try:
            self._hid_device = hid.Device(vid=0x2752, pid=0x0043)
            # self._hid_device.open(vid, pid)
        except:
            # Throw error here
            raise RuntimeError(f"HID device failed to open vid: {vid}, pid: {pid}")

    def write(self, command):
        # init hid command to report id (0) plus 64 0xFF
        hid_buf = [0x00] + [0xFF] * 64
        # Add header specifying command length (+1 for CRC8 byte)
        command = [len(command) + 1] + command
        # Add a CRC8 byte at the end
        command = command + [sum(command) % 0x100]
        # Insert the fully fledged command into the data sequence
        hid_buf[1 : len(command) + 1] = command

        # Send it
        try:
            self._hid_device.write(hid_buf)
        except:
            raise RuntimeError(f"HID send failed!!!")

        # Read back the response
        resp = self._hid_device.read(64)
        # print(f"directly from response: {resp}")

        # First byte is the response length
        resp = resp[1:]
        return resp
