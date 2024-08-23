import threading
import numpy as np
import transforms3d as t3d

from pythonosc import dispatcher
from pythonosc import osc_server

config = {"messages": [],
          "data": [],
          "ip": "127.0.0.1",
          "port": 9007}

class MotionReceiver():
    
    def __init__(self, config):
        
        self.messages = config["messages"]
        self.data = config["data"]
        
        self.ip = config["ip"]
        self.port = config["port"]
        
        self.dispatcher = dispatcher.Dispatcher()
        self.motion_data = {}
        
        for message in self.messages:
            self.dispatcher.map(message, self.receive)
            self.motion_data["message"] = None
            
        self.server = osc_server.ThreadingOSCUDPServer((self.ip, self.port), self.dispatcher)

    def start_server(self):
        self.server.serve_forever()

    def start(self):
        
        self.th = threading.Thread(target=self.start_server)
        self.th.start()
        
    def stop(self):
        self.server.server_close()            
            
    def receive(self, address, *args):
        
        values = np.array(args)
        data_index = self.messages.index(address)
        data_shape = self.data[data_index].shape
        
        values = np.reshape(values,data_shape)

        np.copyto(self.data[data_index], values )