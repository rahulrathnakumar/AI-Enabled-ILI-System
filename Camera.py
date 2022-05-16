import pyrealsense2 as rs        
import time
import json
import atexit


'''
Required class functionality:
Camera class should create an object that can stream depth data and relevant params in real-time.
'''

class Camera(rs.pipeline, rs.config, rs.rs400_advanced_mode):
    '''
    '''
    def __init__(self, preset = None, intrinsics = None, params = None) -> None:
        '''
        Multifunctional camera wrapper class for D435i sensor
        Stream data and params in real-time. 
        Future:
        Custom Filters
        
        Initialize Camera with inherited rs.pipeline and rs.config base classes
        Example: 


        # Additional params
        intrinsics: Intrinsic matrix
        quality params: User-defined parameter (dict) for depth quality metrics

        '''
        rs.pipeline.__init__(self)
        rs.config.__init__(self)
        ctx = rs.context()
        ds5_dev = rs.device()
        devices = ctx.query_devices()
        assert len(list(devices)) == 1, "Either no camera or more than one camera has been connected."
        self.device = devices[0] # Ttemporal_filter.set_option(rs.option.filter_smooth_delta, 0.25)his gives back the sensor options
        self.depth_sensor = self.device.first_depth_sensor()
        self.color_sensor = self.device.first_color_sensor()
        self.motion_sensor = self.device.first_motion_sensor()
        rs.rs400_advanced_mode.__init__(self, devices[0])
        # Parameter attribute for depth sensor
        self.param_dict = self.get_depth_sensor_params()
    def start_camera(self, height = 1280, width = 720, framerate = 30, streams = None):
        '''
        Start camera streams specified by list streams:
        Arguments: 
        height: int, Default : 1280
        width: int, Default : 720
        framerate: int, Default: 30
        streams: List, Default: None
        Optional - Enables motion, color, infrared and depth streams by default for the D435i
        (To be expanded for Lidar and T265 Camera)
        Returns: None
        '''
        if streams == None:
            self.enable_stream(rs.stream.depth, height, width, rs.format.z16, framerate)
            self.enable_stream(rs.stream.color, height, width, rs.format.rgb8, framerate)
            self.enable_stream(rs.stream.infrared, height, width, framerate = framerate)
        else:
            print("NotImplemented.")
        self.start(self)
    def stop_camera(self):
        self.stop()
    def get_depth_sensor_params(self, params = None):
        '''
        Streams all relevant parameters. [Expandable]
        Current parameter list:
        1. Depth Units : self.device.first_depth_sensor.get_option(rs.option.depth_units)
        2. 
        '''
        # Depth sensor options - Basic
        depth_units = self.depth_sensor.get_option(rs.option.depth_units)
        exposure = self.depth_sensor.get_option(rs.option.exposure)
        gain = self.depth_sensor.get_option(rs.option.gain)
        laser_power = self.depth_sensor.get_option(rs.option.laser_power)
        # emitter_enabled = self.depth_sensor.get_option(rs.option.emitter_enabled)
        basic_options_dict = dict({'depthUnits': depth_units, 'exposure': exposure, 'gain': gain, 'laserPower': laser_power})
        # Depth sensor - advanced 
        depth_control = self.get_depth_control() # object depth_control of class rs.STDepthControlGroup. Access attributes separately?
        depth_control_attrs = [d for d in dir(depth_control) if not d.startswith('__')]
        depth_control_vals = [depth_control.__getattribute__(d) for d in depth_control_attrs]
        depth_control_dict = {'STDepthControlGroup': {depth_control_attrs[i]: depth_control_vals[i] for i in range(len(depth_control_attrs))}}
        depth_table = self.get_depth_table() 
        depth_table_attrs = [d for d in dir(depth_table) if not d.startswith('__')]
        depth_table_vals = [depth_table.__getattribute__(d) for d in depth_table_attrs]
        depth_table_dict = {'STDepthTableControl': {depth_table_attrs[i]: depth_table_vals[i] for i in range(len(depth_table_attrs))}}
        # Param dict
        param_dict = basic_options_dict
        # param_dict.update(basic_options_dict)
        param_dict.update(depth_control_dict)
        param_dict.update(depth_table_dict)
        return param_dict
    def set_depth_sensor_params(self, param_dict, get = True):
        '''
        Update depth sensor with new params

        Input: params: Dict of updated params
        Return: None or params depending on get
        '''
        # Basic param update
        self.depth_sensor.set_option(rs.option.depth_units, param_dict[''])
        # Advanced param groups update
        # First update rs 
    def load_preset(self, file):
        jsonObj = json.load(open(file))
        json_string = str(jsonObj).replace("'", '\"')
        self.load_json(json_string)
    def save_json_file(self, params = None):
        print("NotImplemented.")
    # Helpers
    def __get_camera_info__(self, arg):
        return rs.camera_info(arg)
