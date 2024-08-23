import numpy as np
import analysis as ma


class MotionPipeline():
    def __init__(self, oscReceiver, jointWeigths, updateInterval):
            
        self.pos = oscReceiver.data[0]
        self.rot = oscReceiver.data[1]
        self.jointWeigths = jointWeigths
        self.updateInterval = updateInterval
        self.jointCount = len(self.jointWeigths)
        
        # scale value (from cm to m)
        self.posScale = 0.01
        self.posScaled = np.zeros_like(self.pos )
        
        # smooth
        self.posSmoothFactor = 0.9
        self.velSmoothFactor = 0.9
        self.accelSmoothFactor = 0.9
        self.jerkSmoothFactor = 0.9
        
        self.posSmooth = np.zeros_like(self.pos )
        self.velSmooth = np.zeros_like(self.pos )
        self.accelSmooth = np.zeros_like(self.pos )
        self.jerkSmooth = np.zeros_like(self.pos )
        
        # derivatives
        self.vel = np.zeros_like(self.pos)
        self.accel= np.zeros_like(self.pos)
        self.jerk = np.zeros_like(self.pos)
        
        # scalar
        self.posScalar = np.zeros(self.jointCount)
        self.velScalar = np.zeros(self.jointCount)
        self.accelScalar = np.zeros(self.jointCount)
        self.jerkScalar = np.zeros(self.jointCount)

        #quom
        self.quom = np.array([0.0])
        
        # bbox
        self.bbox = np.zeros([2, 3])
        
        # bsphere
        self.bsphere = np.array([4])
        
        # ring buffers
        self.ringSize = 25
        self.posRing = np.zeros([self.ringSize, self.jointCount, 3])
        self.velScalarRing = np.zeros([self.ringSize, self.jointCount])
        self.accelScalarRing = np.zeros([self.ringSize, self.jointCount])
        self.jerkScalarRing = np.zeros([self.ringSize, self.jointCount])
        
        # Laban Effort Factors
        self.windowLength = self.ringSize - 1
        self.flowEffort = np.array([0])
        self.timeEffort = np.array([0])
        self.weightEffort = np.array([0])
        self.timeEffort = np.array([0])
        
    def setUpdateInterval(self, updateInterval):
        self.updateInterval = updateInterval

        
    def update(self):
        
        #print("DataPipeline update")
        #print("self.input_pos_data", self.input_pos_data )

        # pos scale
        _posScaled = self.pos * self.posScale
        
        # pos smooth
        _posSmooth = self.posSmooth * self.posSmoothFactor + _posScaled * (1.0 - self.posSmoothFactor)
        
        # velocity 
        _vel = (_posSmooth - self.posSmooth) / self.updateInterval
        
        # velocity smooth
        _velSmooth = self.velSmooth * self.velSmoothFactor + _vel * (1.0 - self.velSmoothFactor)
        
        # acceleration
        _accel = (_velSmooth - self.velSmooth) / self.updateInterval
        
        # acceleration smooth
        _accelSmooth = self.accelSmooth * self.accelSmoothFactor + _accel * (1.0 - self.accelSmoothFactor)
        
        # jerk
        _jerk = (_accelSmooth - self.accelSmooth) / self.updateInterval
        
        # jerk smooth
        _jerkSmooth = self.jerkSmooth * self.jerkSmoothFactor + _jerk * (1.0 - self.jerkSmoothFactor)  
        
        # pos scalar
        _pos_scalar = np.linalg.norm(_posSmooth, axis=-1)
        
        # vel scalar
        _vel_scalar = np.linalg.norm(_velSmooth, axis=-1)
        
        # accel scalar
        _accel_scalar = np.linalg.norm(_accelSmooth, axis=-1)
        
        # jerk scalar
        _jerk_scalar = np.linalg.norm(_jerkSmooth, axis=-1)

        # quom
        _qom = np.array([np.sum(_vel_scalar * self.jointWeigths) / np.sum(self.jointWeigths)])
        
        # bbox
        _bbox = ma.bounding_box_rt(_posSmooth)

        # bsphere
        _bsphere = ma.bounding_sphere_rt(_posSmooth)

        # ring buffers
        _posRing = np.roll(self.posRing, shift=1, axis=0)
        _posRing[0] = _posSmooth
        
        _velScalarRing = np.roll(self.velScalarRing, shift=1, axis=0)
        _velScalarRing[0] = _vel_scalar

        _accelScalarRing = np.roll(self.accelScalarRing, shift=1, axis=0)
        _accelScalarRing[0] = _accel_scalar
        
        _jerkScalarRing = np.roll(self.jerkScalarRing, shift=1, axis=0)
        _jerkScalarRing[0] = _jerk_scalar
        
        # Laban Effort Factors
        _flow_effort = ma.flow_effort_rt(_jerkScalarRing, self.jointWeigths)
        _time_effort = ma.time_effort_rt(_accelScalarRing, self.jointWeigths)
        _weight_effort = ma.weight_effort_rt(_velScalarRing, self.jointWeigths)
        _space_effort = ma.space_effort_v2_rt(_posRing, self.jointWeigths)
        
        # update all member variables
        self.posScaled = _posScaled
        self.posSmooth = _posSmooth
        self.vel = _vel
        self.velSmooth = _velSmooth
        self.accel = _accel
        self.accelSmooth = _accelSmooth
        self.jerk = _jerk
        self.jerkSmooth = _jerkSmooth
        self.pos_scalar = _pos_scalar
        self.vel_scalar = _vel_scalar
        self.accel_scalar = _accel_scalar
        self.jerk_scalar = _jerk_scalar
        self.qom = _qom
        self.bbox = _bbox
        self.bsphere = _bsphere
        self.posRing = _posRing
        self.velScalarRing = _velScalarRing
        self.accelScalarRing = _accelScalarRing
        self.jerkScalarRing = _jerkScalarRing
        self.flow_effort = _flow_effort
        self.time_effort = _time_effort
        self.weight_effort = _weight_effort
        self.space_effort = _space_effort

        
        #print("posScaled ", self.posScaled)
        #print("posSmooth ", self.posSmooth)
        #print("vel ", self.vel)
        #print("velSmooth ", self.velSmooth)
        #print("accel ", self.accel)
        #print("accelSmooth ", self.accelSmooth)
        #print("jerk ", self.jerk)
        #print("jerkSmooth ", self.jerkSmooth)
        #print("pos_scalar ", self.pos_scalar)
        #print("vel_scalar ", self.vel_scalar)
        #print("accel_scalar ", self.accel_scalar)
        #print("jerk_scalar ", self.jerk_scalar)
        #print("qom ", self.qom)
        #print("bbox ", self.bbox)
        #print("bsphere ", self.bsphere)
        #print("posRing ", self.posRing)
        #print("velScalarRing ", self.velScalarRing)
        #print("accelScalarRing ", self.accelScalarRing)
        #print("jerkScalarRing ", self.jerkScalarRing)
        #print("flow_effort ", self.flow_effort)
        #print("time_effort ", self.time_effort)
        #rint("weight_effort ", self.weight_effort)
        #print("space_effort ", self.space_effort)
        
        
        #print("self.posScale ", self.posScale)
        #print("self.posSmooth ", self.posSmooth)
        #print("self.vel ", self.vel)

"""    
class DataProcScale(DataProc):
    
    def __init__(self, name, outputName, valueScale):
        super().__init__(name)
        self.valueScale = valueScale
        
    def process(self):
        
        if len(self.inputProcs) == 0: 
               return
        
        inputData = self.inputProcs[0].data
        
        if len(inputData) == 0:
            return
        
        # take only the first data from potentially longer list of data
        # ignore the remaining data
        inputData = inputData[0]
        inputDataName = inputData.name
        
        valueDim = inputData.valueDim
        valueCount = inputData.valueCount
"""      
        

        