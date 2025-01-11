
import os
import platform

class MacOSConfig:
    def __init__(self):
        self.huggingface_dir = '/Users/didi/Desktop/KYCode/huggingface'

class WindowsConfig:
    def __init__(self):
        self.huggingface_dir = 'E:\\ARearchCode\\huggingface'

class LinuxConfig:
    def __init__(self):
        pass
        # self.huggingface_dir = 'E:\\ARearchCode\\huggingface'

def get_os():
    system = platform.system()
    if system == 'Windows':
        return 'Windows'
    elif system == 'Darwin':
        return 'macOS'
    elif system == 'Linux':
        return 'Linux'
    else:
        return 'Unknown OS'

def get_cfg_by_os():
    system = platform.system()
    if system == 'Windows':
        return WindowsConfig()
    elif system == 'Darwin':
        return MacOSConfig()
    elif system == 'Linux':
        return LinuxConfig()
    else:
        return 'Unknown OS'


if __name__ == '__main__':
    os_name = get_os()
    print(f"You are using {os_name} operating system.")

    a = get_cfg_by_os()
    print(a.huggingface_dir)

    s = os.path.join(a.huggingface_dir, 'bert')
    print(s)