// ============================================
// UNIVERSAL OSC GESTURE DETECTOR
// Works with PdParty, TouchOSC, or any OSC source
// With improved confidence calculation
// ============================================

(
// ============================================
// USER CONFIGURATION - EDIT THESE SETTINGS
// ============================================
(
// Network settings
~oscPort = 57120;                           // Port to receive OSC messages
~oscIP = "0.0.0.0";                         // IP to listen on (0.0.0.0 = all interfaces)
~controlPort = 57121;  // Port for test control messages


// OSC address to listen to
~oscAddress = '/pdparty/accelerate';        // Change to your OSC address

// Input configuration
~inputNames = ['x', 'y', 'z'];             // Names for your inputs
~numInputs = 3;                             // Number of inputs (auto-detected if different)

// Gesture detection settings
~recognitionThreshold = 30.0;              // Distance threshold (lower = stricter)
~confidenceThreshold = 0.35;               // Minimum confidence to trigger (0.0 to 1.0)
~detectionCooldown = 1.0;                  // Seconds between detections
~maxTemplatesPerGesture = 5;               // Max recordings per gesture

// Motion detection settings
~motionSensitivity = 0.02;                 // Motion threshold (lower = more sensitive)
~stillnessDuration = 0.3;                  // Seconds of stillness to end gesture
~minGestureFrames = 8;                     // Minimum frames for valid gesture
~maxGestureFrames = 100;                   // Maximum frames per gesture

// Display settings
~showAllConfidences = true;                // Always show all gesture confidences
~debugMode = false;                        // Show debug information
);

// ============================================
// GESTURE DETECTOR IMPLEMENTATION
// ============================================
(
~universalGestureDetector = (
	// Core configuration from user settings
	oscPort: ~oscPort,
	oscIP: ~oscIP,
	oscAddress: ~oscAddress,
	inputNames: ~inputNames,
	numInputs: ~numInputs,
	recognitionThreshold: ~recognitionThreshold,
	confidenceThreshold: ~confidenceThreshold,
	detectionCooldown: ~detectionCooldown,
	maxTemplatesPerGesture: ~maxTemplatesPerGesture,
	motionSensitivity: ~motionSensitivity,
	stillnessDuration: ~stillnessDuration,
	minGestureFrames: ~minGestureFrames,
	maxGestureFrames: ~maxGestureFrames,
	showAllConfidences: ~showAllConfidences,
	debugMode: ~debugMode,

	// Core state
	gestureLibrary: Dictionary.new,
	currentGesture: List.new,
	isRecording: false,
	isRecognizing: false,
	gestureCallbacks: Dictionary.new,
	lastDetectionTime: 0,

	// Data processing
	lastInputData: nil,
	lastVelocity: nil,
	lastAcceleration: nil,
	inputBuffer: List.new,

	// Motion detection
	motionHistory: List.new,
	motionHistorySize: 5,
	isInGesture: false,
	gestureStartTime: 0,
	lastMotionTime: 0,

	// Recording state
	currentGestureName: nil,
	recordingFrameCount: 0,

	// Template management
	useTemplateAveraging: true,
	averagedTemplates: Dictionary.new,

	// OSC Setup
	setupOSC: { |self, port, ip, address|
		if(port.notNil, { self.oscPort = port });
		if(ip.notNil, { self.oscIP = ip });
		if(address.notNil, { self.oscAddress = address });

		// Open UDP port
		thisProcess.openUDPPort(self.oscPort);

		// Free existing OSC def
		OSCdef(\universalGesture).free;

		// Create new OSC def
		OSCdef(\universalGesture, { |msg|
			var values, numValues;

			// Extract values (skip the address at index 0)
			values = msg.copyRange(1, msg.size-1);
			numValues = values.size;

			// Auto-detect number of inputs if needed
			if(self.numInputs != numValues, {
				self.numInputs = numValues;
				self.inputNames = numValues.collect { |i| ("input_" ++ (i+1)).asSymbol };
				if(self.debugMode, {
					("Auto-detected " ++ numValues ++ " inputs").postln;
				});
			});

			// Process the frame
			self.processFrame(values);

		}, self.oscAddress, recvPort: self.oscPort);

		("OSC setup complete:").postln;
		("  Port: " ++ self.oscPort).postln;
		("  Address: " ++ self.oscAddress).postln;
		("  Expected inputs: " ++ self.numInputs ++ " (" ++ self.inputNames ++ ")").postln;
	},

	// Calculate motion from input data
	calculateMotion: { |self, inputData|
		var motion, velocity, acceleration, jerk;
		var deltas, velocityDeltas, accelDeltas;
		var velocityMag, accelMag, jerkMag;
		var result;

		result = if(self.lastInputData.isNil, {
			self.lastInputData = inputData.copy;
			self.lastVelocity = Array.fill(self.numInputs, 0);
			self.lastAcceleration = Array.fill(self.numInputs, 0);
			0.0
		}, {
			// Calculate velocity (first derivative)
			deltas = inputData.collect { |val, i|
				val - self.lastInputData[i]
			};
			velocity = deltas;

			// Calculate acceleration (second derivative)
			velocityDeltas = velocity.collect { |val, i|
				val - self.lastVelocity[i]
			};
			acceleration = velocityDeltas;

			// Calculate jerk (third derivative)
			accelDeltas = acceleration.collect { |val, i|
				val - self.lastAcceleration[i]
			};
			jerk = accelDeltas;

			// Calculate magnitudes
			velocityMag = velocity.collect(_.squared).sum.sqrt;
			accelMag = acceleration.collect(_.squared).sum.sqrt;
			jerkMag = jerk.collect(_.squared).sum.sqrt;

			// Weighted combination
			motion = (velocityMag * 1.0) + (accelMag * 0.5) + (jerkMag * 0.2);

			// Store for next frame
			self.lastInputData = inputData.copy;
			self.lastVelocity = velocity.copy;
			self.lastAcceleration = acceleration.copy;

			motion
		});

		result
	},

	// Process each frame
	processFrame: { |self, inputData|
		var motion, currentTime, avgMotion, isSignificantMotion;

		currentTime = SystemClock.seconds;
		motion = self.calculateMotion(inputData);

		// Manual recording mode
		if(self.isRecording, {
			self.currentGesture.add(inputData);
			self.recordingFrameCount = self.recordingFrameCount + 1;

			if(self.debugMode && ((self.recordingFrameCount % 20) == 0), {
				("Recording frame " ++ self.currentGesture.size).postln;
			});

			if(self.currentGesture.size > self.maxGestureFrames, {
				("Max frames reached, stopping recording").postln;
				self.stopRecording(self.currentGestureName);
			});
		});

		// Automatic gesture detection during recognition
		if(self.isRecognizing && self.isRecording.not, {
			// Update motion history
			self.motionHistory.add(motion);
			if(self.motionHistory.size > self.motionHistorySize, {
				self.motionHistory.removeAt(0);
			});

			// Calculate average motion
			avgMotion = if(self.motionHistory.size > 0, {
				self.motionHistory.sum / self.motionHistory.size
			}, {
				motion
			});

			isSignificantMotion = avgMotion > self.motionSensitivity;

			if(isSignificantMotion, {
				if(self.isInGesture.not, {
					// Start new gesture
					self.isInGesture = true;
					self.gestureStartTime = currentTime;
					self.inputBuffer = List.new;

					if(self.debugMode, {
						("GESTURE START - motion: " ++ avgMotion.round(0.003)).postln;
					});
				});

				// Add frame to current gesture
				self.inputBuffer.add(inputData);

				if(self.inputBuffer.size > self.maxGestureFrames, {
					self.recognizeCompleteGesture();
					self.isInGesture = false;
				});

				self.lastMotionTime = currentTime;

			}, {
				// Low motion - check for gesture end
				if(self.isInGesture, {
					var timeSinceLastMotion = currentTime - self.lastMotionTime;

					if((timeSinceLastMotion > self.stillnessDuration) &&
					   (self.inputBuffer.size >= self.minGestureFrames), {
						if(self.debugMode, {
							("GESTURE END - " ++ self.inputBuffer.size ++ " frames").postln;
						});
						self.recognizeCompleteGesture();
						self.isInGesture = false;
					}, {
						if((timeSinceLastMotion > (self.stillnessDuration * 3)) &&
						   (self.inputBuffer.size < self.minGestureFrames), {
							if(self.debugMode, {
								("GESTURE ABANDONED - too short").postln;
							});
							self.isInGesture = false;
						});
					});
				});
			});
		});
	},

	// Start recording
	startRecording: { |self, gestureName|
		if(gestureName.isNil, {
			"ERROR: Gesture name required".postln;
		}, {
			self.currentGestureName = gestureName;
			self.currentGesture = List.new;
			self.recordingFrameCount = 0;
			self.isRecording = true;
			("Recording gesture: " ++ gestureName).postln;
		})
	},

	// Stop recording
	stopRecording: { |self, gestureName|
		var gestureToSave, templates, existingGestures;

		if(self.isRecording && (self.currentGesture.size >= self.minGestureFrames), {
			gestureToSave = self.currentGesture.asArray;

			// Get or create template list
			existingGestures = self.gestureLibrary.at(gestureName);
			if(existingGestures.isNil, {
				self.gestureLibrary.put(gestureName, []);
				existingGestures = [];
				("Creating new gesture: " ++ gestureName).postln;
			}, {
				("Adding to existing gesture: " ++ gestureName).postln;
			});

			templates = existingGestures ++ [gestureToSave];

			// Limit number of templates
			if(templates.size > self.maxTemplatesPerGesture, {
				templates = templates.copyRange(
					templates.size - self.maxTemplatesPerGesture,
					templates.size - 1
				);
			});

			self.gestureLibrary.put(gestureName, templates);

			// Clear averaged template cache
			if(self.useTemplateAveraging, {
				self.averagedTemplates.removeAt(gestureName);
			});

			self.isRecording = false;

			("Gesture '" ++ gestureName ++ "' saved - template " ++
			 templates.size ++ "/" ++ self.maxTemplatesPerGesture ++
			 " with " ++ gestureToSave.size ++ " frames").postln;

		}, {
			if(self.isRecording, {
				("Gesture too short: " ++ self.currentGesture.size ++
				 " frames (min: " ++ self.minGestureFrames ++ ")").postln;
			}, {
				"Not recording".postln;
			});
			self.isRecording = false;
		});

		nil
	},

	// DTW distance calculation
	dtwDistance: { |self, seq1, seq2|
		var n, m, dtwMatrix, cost, minCost;
		var result;

		n = seq1.size;
		m = seq2.size;

		result = if((n == 0) || (m == 0), {
			inf
		}, {
			// Early exit for very different lengths
			if((max(n, m) / min(n, m).max(1)) > 4.0, {
				self.recognitionThreshold * 2
			}, {
				dtwMatrix = Array2D.new(n, m);

				// Initialize first cell
				dtwMatrix.put(0, 0, self.frameDistance(seq1[0], seq2[0]));

				// Fill first row
				(1..(m-1)).do { |j|
					cost = self.frameDistance(seq1[0], seq2[j]);
					dtwMatrix.put(0, j, dtwMatrix.at(0, j-1) + cost);
				};

				// Fill first column
				(1..(n-1)).do { |i|
					cost = self.frameDistance(seq1[i], seq2[0]);
					dtwMatrix.put(i, 0, dtwMatrix.at(i-1, 0) + cost);
				};

				// Fill the matrix
				(1..(n-1)).do { |i|
					(1..(m-1)).do { |j|
						cost = self.frameDistance(seq1[i], seq2[j]);
						minCost = [
							dtwMatrix.at(i-1, j),
							dtwMatrix.at(i, j-1),
							dtwMatrix.at(i-1, j-1)
						].minItem;
						dtwMatrix.put(i, j, cost + minCost);
					};
				};

				dtwMatrix.at(n-1, m-1) / max(n, m)
			})
		});

		result
	},

	// Distance between two frames
	frameDistance: { |self, frame1, frame2|
		var sum = 0;
		self.numInputs.do { |i|
			if((frame1.size > i) && (frame2.size > i), {
				var diff = frame1[i] - frame2[i];
				sum = sum + (diff * diff);
			});
		};
		sum.sqrt
	},

	// Average multiple templates
	averageTemplates: { |self, templates|
		var avgTemplate, targetLen, numTemplates;
		var result;

		numTemplates = templates.size;
		result = if(numTemplates == 0, {
			[]
		}, {
			if(numTemplates == 1, {
				templates[0]
			}, {
				// Use median length
				targetLen = templates.collect(_.size).sort[(numTemplates/2).asInteger];

				// Resample all templates to target length
				templates = templates.collect { |template|
					self.resampleSequence(template, targetLen)
				};

				// Average the resampled templates
				avgTemplate = Array.fill(targetLen, { |frameIdx|
					var frameSum = Array.fill(self.numInputs, 0);

					templates.do { |template|
						self.numInputs.do { |inputIdx|
							frameSum[inputIdx] = frameSum[inputIdx] + template[frameIdx][inputIdx];
						};
					};

					frameSum.collect(_ / numTemplates)
				});

				avgTemplate
			})
		});

		result
	},

	// Resample sequence to target length
	resampleSequence: { |self, sequence, targetLen|
		var sourceLen = sequence.size;
		var resampled;
		var result;

		result = if(sourceLen == targetLen, {
			sequence
		}, {
			resampled = Array.fill(targetLen, { |i|
				var sourceIdx = (i * (sourceLen - 1) / (targetLen - 1));
				var alpha = sourceIdx.frac;
				var idx1 = sourceIdx.floor.asInteger;
				var idx2 = (idx1 + 1).min(sourceLen - 1);

				if(idx1 >= (sourceLen - 1), {
					sequence.last
				}, {
					// Linear interpolation
					Array.fill(self.numInputs, { |j|
						(sequence[idx1][j] * (1 - alpha)) + (sequence[idx2][j] * alpha)
					})
				})
			});

			resampled
		});

		result
	},

	// Recognize completed gesture with improved confidence calculation
	recognizeCompleteGesture: { |self|
		var candidates, distance, gestureName, confidence, bestMatch;
		var currentTime, gestureSequence, bestDistance, allConfidences;
		var testTemplate, allDistances, minDistance, maxDistance, normalizedDistance;
		var totalConf;

		currentTime = SystemClock.seconds;

		// Check cooldown
		if((currentTime - self.lastDetectionTime) < self.detectionCooldown, {
			if(self.debugMode, {
				("In cooldown period").postln;
			});
			nil
		}, {
			if(self.gestureLibrary.size == 0, {
				nil
			}, {
				gestureSequence = self.inputBuffer.asArray;

				if(gestureSequence.size < self.minGestureFrames, {
					nil
				}, {
					candidates = [];
					allConfidences = Dictionary.new;
					allDistances = Dictionary.new;

					// Test each gesture
					self.gestureLibrary.keysValuesDo { |name, templates|
						bestDistance = inf;

						// Use averaged template if enabled
						testTemplate = if(self.useTemplateAveraging, {
							var avgTemplate = self.averagedTemplates.at(name);
							if(avgTemplate.isNil, {
								avgTemplate = self.averageTemplates(templates);
								self.averagedTemplates.put(name, avgTemplate);
							});
							[avgTemplate]
						}, {
							templates
						});

						// Test against template(s)
						testTemplate.do { |template|
							if(template.size >= self.minGestureFrames, {
								try {
									distance = self.dtwDistance(gestureSequence, template);
									if(distance < bestDistance, {
										bestDistance = distance;
									});
								} { |error|
									if(self.debugMode, {
										("Error testing " ++ name ++ ": " ++ error.errorString).postln;
									});
								};
							});
						};

						// Store distance for this gesture
						allDistances.put(name, bestDistance);

						// Add to candidates if within threshold
						if(bestDistance < self.recognitionThreshold, {
							candidates = candidates ++ [[name, bestDistance]];
						});
					};

					// Calculate improved confidences
					if(allDistances.size > 0, {
						// Find min and max distances for normalization
						minDistance = allDistances.values.minItem;
						maxDistance = allDistances.values.maxItem;

						// Calculate confidence for each gesture
						allDistances.keysValuesDo { |name, dist|
							var conf;

							if(dist >= self.recognitionThreshold, {
								// Beyond threshold - very low confidence
								conf = max(0, 0.1 - ((dist - self.recognitionThreshold) / self.recognitionThreshold));
								conf = conf.max(0);
							}, {
								// Within threshold - calculate confidence
								if(minDistance == maxDistance, {
									// All distances are the same (unlikely)
									conf = 0.5;
								}, {
									// Exponential scaling for better differentiation
									normalizedDistance = (dist - minDistance) / (maxDistance - minDistance);
									// Use exponential function for sharper differences
									conf = exp(normalizedDistance.neg * 3) * (1 - (dist / self.recognitionThreshold));
									conf = conf.clip(0, 1);
								});
							});

							allConfidences.put(name, conf);
						};

						// Normalize confidences so they sum to 1 (like probabilities)
						totalConf = allConfidences.values.sum;
						if(totalConf > 0, {
							allConfidences.keysValuesDo { |name, conf|
								allConfidences.put(name, conf / totalConf);
							};
						});
					});

					// Show enhanced gesture analysis if enabled
					if(self.showAllConfidences && (allConfidences.size > 0), {
						("--- Gesture Analysis ---").postln;
						allDistances.keysValuesDo { |name, dist|
							var conf = allConfidences.at(name);
							var status = if(dist < self.recognitionThreshold, {"✓"}, {"✗"});
							("  " ++ status ++ " " ++ name ++ ": " ++
							 (conf * 100).round(0.1) ++ "% (dist: " ++
							 dist.round(0.01) ++ ")").postln;
						};
					});

					// Process best match
					if(candidates.size > 0, {
						candidates = candidates.sort { |a, b| a[1] < b[1] };
						bestMatch = candidates[0];
						gestureName = bestMatch[0];
						distance = bestMatch[1];
						confidence = allConfidences.at(gestureName);

						if(confidence > self.confidenceThreshold, {
							self.lastDetectionTime = currentTime;
							(">>> DETECTED: " ++ gestureName ++
							 " (confidence: " ++ (confidence * 100).round(0.1) ++ "%) <<<").postln;

							// Call callback if exists
							if(self.gestureCallbacks.at(gestureName).notNil, {
								try {
									self.gestureCallbacks.at(gestureName).(confidence);
								} { |error|
									("Callback error: " ++ error.errorString).postln;
								};
							});
						}, {
							if(self.debugMode, {
								("Best match below confidence threshold: " ++ gestureName ++
								 " at " ++ (confidence * 100).round(0.1) ++ "%").postln;
							});
						});
					}, {
						if(self.debugMode, {
							("No gestures matched (all distances above threshold " ++
							 self.recognitionThreshold ++ ")").postln;
						});
					});
				});
			});
		});
	},

	// Start recognition
	startRecognition: { |self|
		if(self.gestureLibrary.size == 0, {
			"No gestures in library - record some gestures first!".postln;
		}, {
			self.isRecognizing = true;
			self.isInGesture = false;
			self.inputBuffer = List.new;

			("Recognition started with " ++ self.gestureLibrary.size ++ " gestures").postln;
			("Threshold: " ++ self.recognitionThreshold).postln;
			("Confidence: " ++ (self.confidenceThreshold * 100) ++ "%").postln;
		})
	},

	// Stop recognition
	stopRecognition: { |self|
		self.isRecognizing = false;
		self.isInGesture = false;
		"Recognition stopped".postln;
	},

	// Add callback for gesture
	addCallback: { |self, gestureName, callback|
		self.gestureCallbacks.put(gestureName, callback);
		("Callback added for gesture: " ++ gestureName).postln;
	},

	// Configuration methods
	setRecognitionThreshold: { |self, threshold|
		self.recognitionThreshold = threshold.clip(1.0, 100.0);
		("Recognition threshold: " ++ self.recognitionThreshold).postln;
	},

	setConfidenceThreshold: { |self, threshold|
		self.confidenceThreshold = threshold.clip(0.0, 1.0);
		("Confidence threshold: " ++ (self.confidenceThreshold * 100) ++ "%").postln;
	},

	setMotionSensitivity: { |self, sensitivity|
		self.motionSensitivity = sensitivity.clip(0.001, 1.0);
		("Motion sensitivity: " ++ self.motionSensitivity).postln;
	},

	setStillnessDuration: { |self, duration|
		self.stillnessDuration = duration.clip(0.1, 5.0);
		("Stillness duration: " ++ self.stillnessDuration ++ "s").postln;
	},

	setDebugMode: { |self, enabled|
		self.debugMode = enabled;
		("Debug mode: " ++ enabled).postln;
	},

	setShowAllConfidences: { |self, enabled|
		self.showAllConfidences = enabled;
		("Show all confidences: " ++ enabled).postln;
	},

	// Utility methods
	listGestures: { |self|
		if(self.gestureLibrary.size > 0, {
			"=== Gesture Library ===".postln;
			self.gestureLibrary.keysValuesDo { |name, templates|
				var frameCounts = templates.collect(_.size);
				("  " ++ name ++ ": " ++ templates.size ++
				 " templates (" ++ frameCounts ++ " frames)").postln;
			};
		}, {
			"Gesture library is empty".postln;
		});
	},

	clearLibrary: { |self|
		self.gestureLibrary = Dictionary.new;
		self.gestureCallbacks = Dictionary.new;
		self.averagedTemplates = Dictionary.new;
		"Library cleared".postln;
	},

	removeGesture: { |self, gestureName|
		if(self.gestureLibrary.at(gestureName).notNil, {
			self.gestureLibrary.removeAt(gestureName);
			self.gestureCallbacks.removeAt(gestureName);
			self.averagedTemplates.removeAt(gestureName);
			("Gesture '" ++ gestureName ++ "' removed").postln;
		}, {
			("Gesture '" ++ gestureName ++ "' not found").postln;
		});
	},

	getStatus: { |self|
		"=== Gesture Detector Status ===".postln;
		("OSC Port: " ++ self.oscPort).postln;
		("OSC Address: " ++ self.oscAddress).postln;
		("Number of inputs: " ++ self.numInputs).postln;
		("Recording: " ++ self.isRecording).postln;
		("Recognizing: " ++ self.isRecognizing).postln;
		("In gesture: " ++ self.isInGesture).postln;
		("Gestures in library: " ++ self.gestureLibrary.size).postln;
		("Recognition threshold: " ++ self.recognitionThreshold).postln;
		("Motion sensitivity: " ++ self.motionSensitivity).postln;
		("Debug mode: " ++ self.debugMode).postln;
	},

	// Cleanup
	cleanup: { |self|
		self.stopRecognition();
		self.isRecording = false;
		OSCdef(\universalGesture).free;
		"Cleaned up".postln;
	},

	// Initialize
	init: { |self, port, ip, address|
		self.cleanup();
		self.setupOSC(port, ip, address);

		// Initialize motion tracking
		self.motionHistory = List.new;
		self.lastInputData = nil;
		self.lastVelocity = nil;
		self.lastAcceleration = nil;

		"Universal Gesture Detector initialized!".postln;
	}
);

// Initialize the detector
~universalGestureDetector.init(~oscPort, ~oscIP, ~oscAddress);
);

// ============================================
// USAGE EXAMPLES
// ============================================
(
"".postln;
"=== UNIVERSAL OSC GESTURE DETECTOR READY ===".postln;
"".postln;
"CURRENT CONFIGURATION:".postln;
("  OSC Port: " ++ ~oscPort).postln;
("  OSC Address: " ++ ~oscAddress).postln;
("  Input names: " ++ ~inputNames).postln;
("  Recognition threshold: " ++ ~recognitionThreshold).postln;
"".postln;
"QUICK START:".postln;
"".postln;
"// 1. Record gestures:".postln;
"~universalGestureDetector.startRecording(\"shake\");".postln;
"// ... perform gesture ...".postln;
"~universalGestureDetector.stopRecording(\"shake\");".postln;
"".postln;
"// 2. Record more gestures:".postln;
"~universalGestureDetector.startRecording(\"tilt\");".postln;
"// ... perform gesture ...".postln;
"~universalGestureDetector.stopRecording(\"tilt\");".postln;
"".postln;
"// 3. Start recognition:".postln;
"~universalGestureDetector.startRecognition();".postln;
"".postln;
"CONFIGURATION:".postln;
"~universalGestureDetector.setRecognitionThreshold(40);  // 10-100".postln;
"~universalGestureDetector.setMotionSensitivity(0.01);   // Lower = more sensitive".postln;
"~universalGestureDetector.setDebugMode(true);           // Show debug info".postln;
"~universalGestureDetector.setShowAllConfidences(true);  // Show all %".postln;
"".postln;
"CALLBACKS:".postln;
"~universalGestureDetector.addCallback(\"shake\", { |conf|".postln;
"    (\"Shake detected at \" ++ (conf * 100).round(0.1) ++ \"% confidence!\").postln;".postln;
"    // Trigger your sound/visual here".postln;
"});".postln;
"".postln;
"UTILITIES:".postln;
"~universalGestureDetector.listGestures();   // Show recorded gestures".postln;
"~universalGestureDetector.getStatus();      // Show current status".postln;
"~universalGestureDetector.clearLibrary();   // Clear all gestures".postln;
"".postln;
);


// Setup OSC control port (different from data input port)
thisProcess.openUDPPort(~controlPort);

// OSC to start recording
// Send: /record/start "gestureName"
OSCdef(\startRec, { |msg|
    var gestureName = msg[1].asString;
    if(gestureName.notNil and: { gestureName != "" }, {
        ("OSC: Starting recording - " ++ gestureName).postln;
        ~universalGestureDetector.startRecording(gestureName.asSymbol);
    }, {
        "OSC: ERROR - Gesture name required".postln;
    });
}, '/record/start', recvPort: ~controlPort);

// OSC to stop recording
// Send: /record/stop "gestureName"
OSCdef(\stopRec, { |msg|
    var gestureName = msg[1].asString;
    ("OSC: Stopping recording - " ++ gestureName).postln;
    ~universalGestureDetector.stopRecording(gestureName.asSymbol);
}, '/record/stop', recvPort: ~controlPort);

"".postln;
"=== OSC RECORDING CONTROL READY ===".postln;
("Control Port: " ++ ~controlPort).postln;
"Commands:".postln;
"  /record/start <name>  - Start recording".postln;
"  /record/stop <name>   - Stop recording".postln;
"".postln;


);



(


// ============================================
// CONFIGURATION EXAMPLES FOR DIFFERENT SETUPS
// ============================================

// Example 1: PdParty Accelerometer (3 inputs: x, y, z)
(
~oscPort = 57120;
~oscAddress = '/pdparty/accelerate';
~inputNames = ['x', 'y', 'z'];
~numInputs = 3;
~universalGestureDetector.init(~oscPort, nil, ~oscAddress);
);

// Example 2: TouchOSC XY Pad (2 inputs: x, y)
(
~oscPort = 8000;
~oscAddress = '/1/xy1';
~inputNames = ['x', 'y'];
~numInputs = 2;
~universalGestureDetector.init(~oscPort, nil, ~oscAddress);
);

// Example 3: Custom Multi-parameter Controller (5 inputs)
(
~oscPort = 9000;
~oscAddress = '/controller/data';
~inputNames = ['param1', 'param2', 'param3', 'param4', 'param5'];
~numInputs = 5;
~universalGestureDetector.init(~oscPort, nil, ~oscAddress);
);

// Example 4: Many inputs)
(
~oscPort = 7000;
~oscAddress = '/mocap/markers';
~inputNames = 10.collect { |i| ("marker_" ++ (i+1)).asSymbol };
~numInputs = 10;
~universalGestureDetector.init(~oscPort, nil, ~oscAddress);
);


// -----------------Test sending OSC messages to ourselves----------------------
~test = NetAddr("127.0.0.1", ~controlPort);

// Start recording a "shake" gesture
~test.sendMsg('/record/start', "shake");


// Stop recording (execute this after performing gesture)
~test.sendMsg('/record/stop', "shake");

~universalGestureDetector.startRecognition();


// ============================================
// EXAMPLE: From external OSC source
// ============================================

/*
From PdParty, TouchOSC, or any OSC app:

1. Send to your computer's IP, port 57121

2. To start recording:
   Address: /record/start
   Argument: "shake" (or any gesture name)

3. To stop recording:
   Address: /record/stop
   Argument: "shake" (same gesture name)
*/


// ============================================
// TESTING AND TUNING
// ============================================


// For testing with PdParty accelerometer "MANUALLY"

// Record gestures manually one at a time
~universalGestureDetector.clearLibrary();

// Record first shake sample
~universalGestureDetector.startRecording("shake");
// ... perform shake gesture manually ...
 ~universalGestureDetector.stopRecording("shake");

// Record first tilt sample
~universalGestureDetector.startRecording("tilt");
// ... perform tilt gesture manually ...
 ~universalGestureDetector.stopRecording("tilt");

// When done recording, start recognition
 ~universalGestureDetector.startRecognition();


// -----------------------Automated PRocesss---------------------

(
Routine {
    // Record shake gesture 3 times
    3.do { |i|
        ("Recording shake sample " ++ (i+1)).postln;
        ~universalGestureDetector.startRecording("shake");
        3.wait;
        ~universalGestureDetector.stopRecording("shake");
        1.wait;
    };

    // Record tilt gesture 3 times
    3.do { |i|
        ("Recording tilt sample " ++ (i+1)).postln;
        ~universalGestureDetector.startRecording("tilt");
        3.wait;
        ~universalGestureDetector.stopRecording("tilt");
        1.wait;
    };

    // Start recognition
    ~universalGestureDetector.startRecognition();
    "Ready for gesture detection!".postln;
}.play;
);

// Adjust sensitivity if needed
(
// If gestures are not being detected well
~universalGestureDetector.setRecognitionThreshold(35);  // More lenient
~universalGestureDetector.setConfidenceThreshold(0.3);  // Lower confidence required
~universalGestureDetector.setMotionSensitivity(0.015);  // More sensitive to motion
);

// View current library
~universalGestureDetector.listGestures();

// Check detector status
~universalGestureDetector.getStatus();