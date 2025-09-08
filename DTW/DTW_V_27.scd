(
// ============================================
// OSC CONFIGURATION - EDIT THESE SETTINGS
// ============================================
(
~oscPort = 9006;                    // OSC receive port
~controlPort = 57121;  // Port for control messages

~oscRotationAddress = '/mocap/joint/rot';           // OSC address for rotation data
~oscVelocityAddress = '/mocap/joint/rot_velocity';  // OSC address for velocity data
~oscAccelerationAddress = '/mocap/joint/rot_acceleration'; // OSC address for acceleration data
~oscJerkAddress = '/mocap/joint/rot_jerk';         // OSC address for jerk data
~expectedJoints = 23;               // Expected number of joints (will auto-detect if different)
~recognitionThreshold = 25.0;       // Distance threshold for recognition (higher = more lenient)
~confidenceThreshold = 0.4;         // Minimum confidence to trigger detection (0.0 to 1.0)
~detectionCooldown = 0.5;           // Seconds to wait between detections
~maxTemplatesPerGesture = 5;        // Maximum templates to keep per gesture
);

// ============================================
// GESTURE DETECTOR IMPLEMENTATION
// ============================================
(
~gestureDetector = (
	// Core configuration from global settings
	oscPort: ~oscPort,
	recognitionThreshold: ~recognitionThreshold,
	confidenceThreshold: ~confidenceThreshold,
	maxTemplatesPerGesture: ~maxTemplatesPerGesture,
	detectionCooldown: ~detectionCooldown,

	// Core state
	gestureLibrary: Dictionary.new,
	currentGesture: List.new,
	isRecording: false,
	isRecognizing: false,
	gestureBuffer: Array.fill(100, { Array.fill(276, 0) }),
	gestureCallbacks: Dictionary.new,
	frameSkipCounter: 0,
	frameSkipRate: 8,
	recognitionTask: nil,
	recognitionInterval: 0.2,
	maxGestureLength: 50,
	minGestureLength: 5,
	lastDetectionTime: 0,
	debugMode: false,

	// Multi-joint configuration
	numJoints: ~expectedJoints,
	featuresPerJoint: 12,

	// NEW: Joint subset selection
	useJointSubset: false,
	jointSubset: nil,  // Array of joint indices to use
	jointSubsetMap: Dictionary.new,  // Maps subset index to original joint index
	effectiveNumJoints: ~expectedJoints,  // Number of joints actually being used

	// Motion state tracking
	isInGesture: false,
	gestureStartTime: 0,
	gestureFrames: List.new,
	lastMotionTime: 0,

	// Motion detection parameters
	motionBaseline: 0.0,
	motionHistory: Array.fill(10, 0),
	motionHistoryIdx: 0,
	motionThreshold: 0.008,

	// Speed-invariant parameters
	minGestureMotion: 0.015,
	maxGesturePause: 1.0,
	gestureEndDelay: 0.1,
	velocityDecayThreshold: 0.8,

	// Motion analysis
	currentVelocityMagnitude: 0.0,
	peakVelocityInGesture: 0.0,
	totalGestureMotion: 0.0,
	consecutiveLowMotionFrames: 0,
	gestureMotionProfile: List.new,

	// Gesture quality metrics
	minGestureComplexity: 0.6,
	gestureVelocitySmoothing: 0.5,

	// OSC data storage
	lastOSCData: (
		rot: Array.fill(69, 0),
		rotVel: Array.fill(69, 0),
		rotAcc: Array.fill(69, 0),
		rotJerk: Array.fill(69, 0)
	),

	// NEW: Track actual received data size
	actualDataSize: (
		rot: 0,
		rotVel: 0,
		rotAcc: 0,
		rotJerk: 0
	),

	// Adaptive weighting system
	gestureJointWeights: Dictionary.new,
	adaptiveWeightingEnabled: false,
	weightLearningRate: 0.2,
	minWeight: 0.05,
	maxWeight: 5.0,
	enableWeightCaching: true,
	skipFramesForWeightComputation: 5,
	lastFeatures: nil,
	motionAccumulator: nil,
	recordingFrameCount: 0,

	// Gesture-specific motion analysis
	gestureMotionProfiles: Dictionary.new,
	jointImportanceThreshold: 0.1,
	weightUpdateIterations: 3,
	jointMotionVariance: Dictionary.new,
	crossGestureWeights: Dictionary.new,

	// Template averaging
	useTemplateAveraging: true,
	averagedTemplates: Dictionary.new,

	// NEW: Set which joints to use
	setJointSubset: { |self, jointIndices|
		var maxJoint;

		if(jointIndices.isNil, {
			// Clear subset - use all joints
			self.useJointSubset = false;
			self.jointSubset = nil;
			self.jointSubsetMap.clear();
			self.effectiveNumJoints = self.numJoints;

			// Reset buffer size
			self.gestureBuffer = Array.fill(100, {
				Array.fill(self.effectiveNumJoints * self.featuresPerJoint, 0)
			});

			("Using all " ++ self.numJoints ++ " joints").postln;
		}, {
			// Validate joint indices
			maxJoint = jointIndices.maxItem;
			if(maxJoint >= self.numJoints, {
				("ERROR: Joint index " ++ maxJoint ++ " exceeds available joints (" ++ self.numJoints ++ ")").postln;
				nil;
			}, {
				// Set up joint subset
				self.useJointSubset = true;
				self.jointSubset = jointIndices.asArray.sort;
				self.effectiveNumJoints = self.jointSubset.size;

				// Create mapping
				self.jointSubsetMap.clear();
				self.jointSubset.do { |originalIdx, subsetIdx|
					self.jointSubsetMap.put(subsetIdx, originalIdx);
				};

				// Reset buffer with new size
				self.gestureBuffer = Array.fill(100, {
					Array.fill(self.effectiveNumJoints * self.featuresPerJoint, 0)
				});

				("Using joint subset: " ++ self.jointSubset).postln;
				("Effective joints: " ++ self.effectiveNumJoints).postln;
			});
		});
	},

	// NEW: Set joint range (e.g., joints 0-10)
	setJointRange: { |self, startJoint, endJoint|
		var jointIndices;

		if(startJoint.isNil || endJoint.isNil, {
			self.setJointSubset(nil);  // Clear subset
		}, {
			jointIndices = (startJoint..endJoint);
			self.setJointSubset(jointIndices);
		});
	},

	// Average multiple templates into one representative template
	averageTemplates: { |self, templates|
		var avgTemplate, maxLen, minLen, targetLen, numTemplates;
		var result;

		numTemplates = templates.size;
		result = if(numTemplates == 0, {
			[]
		}, {
			if(numTemplates == 1, {
				templates[0]
			}, {
				minLen = templates.collect(_.size).minItem;
				maxLen = templates.collect(_.size).maxItem;
				targetLen = ((minLen + maxLen) / 2).asInteger;

				templates = templates.collect { |template|
					self.resampleSequence(template, targetLen);
				};

				avgTemplate = Array.fill(targetLen, { |frameIdx|
					var frameSum = Array.fill(self.effectiveNumJoints * self.featuresPerJoint, 0);

					templates.do { |template|
						template[frameIdx].do { |val, idx|
							frameSum[idx] = frameSum[idx] + val;
						};
					};

					frameSum.collect(_ / numTemplates)
				});

				avgTemplate;
			});
		});

		result
	},

	// Resample sequence to target length
	resampleSequence: { |self, sequence, targetLen|
		var sourceLen, resampled, sourceIdx, alpha;
		var result;

		sourceLen = sequence.size;
		result = if(sourceLen == targetLen, {
			sequence
		}, {
			resampled = Array.fill(targetLen, { |i|
				sourceIdx = (i * (sourceLen - 1) / (targetLen - 1));
				alpha = sourceIdx.frac;

				if(sourceIdx.floor >= (sourceLen - 1), {
					sequence.last
				}, {
					var idx1 = sourceIdx.floor.asInteger;
					var idx2 = (idx1 + 1).min(sourceLen - 1);
					var frame1 = sequence[idx1];
					var frame2 = sequence[idx2];

					Array.fill(frame1.size, { |j|
						(frame1[j] * (1 - alpha)) + (frame2[j] * alpha)
					})
				})
			});

			resampled;
		});

		result
	},

	// MODIFIED: Analyze joint importance only for effective joints
	analyzeGestureJointImportance: { |self, gestureName, gestureSequence|
		var jointMotionMagnitudes, jointVariances, jointWeights;
		var totalMotionPerJoint, frameCount, avgMotionPerJoint;
		var motionThreshold, importantJoints, gestureProfile;
		var result;

		result = if(self.adaptiveWeightingEnabled.not, {
			Array.fill(self.effectiveNumJoints, 1.0);
		}, {
			frameCount = gestureSequence.size;
			totalMotionPerJoint = Array.fill(self.effectiveNumJoints, 0);
			jointVariances = Array.fill(self.effectiveNumJoints, 0);

			if(frameCount < 3, {
				Array.fill(self.effectiveNumJoints, 1.0);
			}, {
				self.effectiveNumJoints.do { |jointIdx|
					var jointMotions, mean, variance, baseIdx;

					jointMotions = [];

					gestureSequence.do { |frame|
						baseIdx = jointIdx * self.featuresPerJoint;
						if(frame.size > (baseIdx + 8), {
							var velMag, accMag, motion;

							velMag = (frame.at(baseIdx + 3).squared +
							         frame.at(baseIdx + 4).squared +
							         frame.at(baseIdx + 5).squared).sqrt;

							accMag = (frame.at(baseIdx + 6).squared +
							         frame.at(baseIdx + 7).squared +
							         frame.at(baseIdx + 8).squared).sqrt;

							motion = velMag + (accMag * 0.3);
							jointMotions = jointMotions ++ [motion];
							totalMotionPerJoint.put(jointIdx, totalMotionPerJoint.at(jointIdx) + motion);
						});
					};

					if(jointMotions.size > 0, {
						mean = totalMotionPerJoint.at(jointIdx) / jointMotions.size;
						variance = jointMotions.collect { |m| (m - mean).squared }.sum / jointMotions.size;
						jointVariances.put(jointIdx, variance);
					});
				};

				avgMotionPerJoint = totalMotionPerJoint.collect { |total| total / frameCount };
				motionThreshold = avgMotionPerJoint.sum / self.effectiveNumJoints * 0.3;

				jointWeights = Array.fill(self.effectiveNumJoints, self.minWeight);

				self.effectiveNumJoints.do { |jointIdx|
					var motionWeight, varianceWeight, finalWeight;
					var avgMotion, variance;

					avgMotion = avgMotionPerJoint.at(jointIdx);
					variance = jointVariances.at(jointIdx);

					motionWeight = if(avgMotion > motionThreshold, {
						(avgMotion / motionThreshold).min(self.maxWeight);
					}, {
						self.minWeight;
					});

					varianceWeight = if(variance > 0, {
						1.0 / (1.0 + variance);
					}, {
						1.0;
					});

					finalWeight = (motionWeight * 0.7) + (varianceWeight * 0.3);
					finalWeight = finalWeight.clip(self.minWeight, self.maxWeight);

					jointWeights.put(jointIdx, finalWeight);
				};

				gestureProfile = (
					avgMotionPerJoint: avgMotionPerJoint,
					jointVariances: jointVariances,
					motionThreshold: motionThreshold,
					learnedWeights: jointWeights
				);

				self.gestureMotionProfiles.put(gestureName, gestureProfile);

				if(self.debugMode, {
					("Important joints for '" ++ gestureName ++ "':").postln;
					jointWeights.collect { |weight, idx|
						var originalJointIdx = if(self.useJointSubset, {
							self.jointSubsetMap.at(idx)
						}, {
							idx
						});
						[originalJointIdx, weight]
					}
						.sort { |a, b| a[1] > b[1] }
						.copyRange(0, 4.min(jointWeights.size-1))
						.do { |pair, i|
							("  Joint " ++ pair[0] ++ ": weight " ++ pair[1].round(0.01)).postln;
						};
				});

				jointWeights;
			});
		});

		result
	},

	// MODIFIED: Learn cross-gesture weights for effective joints
	learnCrossGestureWeights: { |self|
		var allGestureProfiles, discriminativeWeights, gestureNames, currentProfile, otherProfiles,
		    discriminativeJointWeights, currentMotion, otherMotions, discriminativeness, avgOtherMotion,
		    difference, maxMotion;
		var result;

		result = if(self.adaptiveWeightingEnabled.not, {
			nil
		}, {
			if(self.gestureMotionProfiles.size < 2, {
				nil;
			}, {
				gestureNames = self.gestureMotionProfiles.keys.asArray;
				discriminativeWeights = Dictionary.new;

				gestureNames.do { |gestureName|
					currentProfile = self.gestureMotionProfiles.at(gestureName);
					otherProfiles = gestureNames.select { |name| name != gestureName }.collect { |name|
						self.gestureMotionProfiles.at(name);
					};

					discriminativeJointWeights = Array.fill(self.effectiveNumJoints, 1.0);

					self.effectiveNumJoints.do { |jointIdx|
						currentMotion = currentProfile.avgMotionPerJoint.at(jointIdx);
						otherMotions = otherProfiles.collect { |profile|
							profile.avgMotionPerJoint.at(jointIdx)
						};

						discriminativeness = if(otherMotions.size > 0, {
							avgOtherMotion = otherMotions.sum / otherMotions.size;
							difference = (currentMotion - avgOtherMotion).abs;
							maxMotion = max(currentMotion, avgOtherMotion);

							if(maxMotion > 0.001, {
								(difference / maxMotion) + 1.0;
							}, {
								1.0;
							});
						}, {
							1.0;
						});

						discriminativeJointWeights.put(jointIdx, discriminativeness.clip(1.0, 3.0));
					};

					discriminativeWeights.put(gestureName, discriminativeJointWeights);
				};

				self.crossGestureWeights = discriminativeWeights;

				if(self.debugMode, {
					("Learned cross-gesture discriminative weights for " ++ gestureNames.size ++ " gestures").postln;
				});

				discriminativeWeights;
			});
		});

		result
	},

	// MODIFIED: Get weights for effective joints
	getGestureWeights: { |self, gestureName|
		var importanceWeights, discriminativeWeights, combinedWeights;
		var result;

		result = if(self.adaptiveWeightingEnabled.not, {
			Array.fill(self.effectiveNumJoints, 1.0);
		}, {
			importanceWeights = self.gestureJointWeights.at(gestureName);
			if(importanceWeights.isNil, {
				importanceWeights = Array.fill(self.effectiveNumJoints, 1.0);
			});

			discriminativeWeights = self.crossGestureWeights.at(gestureName);
			if(discriminativeWeights.isNil, {
				discriminativeWeights = Array.fill(self.effectiveNumJoints, 1.0);
			});

			combinedWeights = Array.fill(self.effectiveNumJoints, 1.0);
			self.effectiveNumJoints.do { |jointIdx|
				var importance = importanceWeights.at(jointIdx);
				var discriminative = discriminativeWeights.at(jointIdx);
				var combined = (importance * 0.7) + (discriminative * 0.3);
				combinedWeights.put(jointIdx, combined.clip(self.minWeight, self.maxWeight));
			};

			combinedWeights;
		});

		result
	},

	// Learn and update weights for a specific gesture
	learnGestureWeights: { |self, gestureName, gestureSequence|
		var jointWeights, existingWeights;
		var result;

		result = if(self.adaptiveWeightingEnabled.not, {
			nil
		}, {
			jointWeights = self.analyzeGestureJointImportance(gestureName, gestureSequence);

			existingWeights = self.gestureJointWeights.at(gestureName);
			if(existingWeights.notNil, {
				var blendedWeights = Array.fill(self.effectiveNumJoints, 1.0);
				self.effectiveNumJoints.do { |jointIdx|
					var existing = existingWeights.at(jointIdx);
					var new = jointWeights.at(jointIdx);
					var blended = (existing * (1.0 - self.weightLearningRate)) + (new * self.weightLearningRate);
					blendedWeights.put(jointIdx, blended);
				};
				self.gestureJointWeights.put(gestureName, blendedWeights);
			}, {
				self.gestureJointWeights.put(gestureName, jointWeights);
			});

			self.learnCrossGestureWeights();

			jointWeights;
		});

		result
	},

	// Update motion baseline continuously
	updateMotionBaseline: { |self, currentMotion|
		var sortedHistory;

		self.motionHistory.put(self.motionHistoryIdx, currentMotion);
		self.motionHistoryIdx = (self.motionHistoryIdx + 1) % self.motionHistory.size;

		sortedHistory = self.motionHistory.copy.sort;
		self.motionBaseline = sortedHistory.at((sortedHistory.size / 2).asInteger);

		self.motionThreshold = self.motionBaseline + 0.015;
	},

	// MODIFIED: Calculate motion only for effective joints
	calculateMotionMagnitude: { |self, features|
		var totalMotion, activeJoints, jointIdx, baseIdx, velMag, accMag, jointMotion;

		totalMotion = 0.0;
		activeJoints = 0;

		self.effectiveNumJoints.do { |ji|
			jointIdx = ji;
			baseIdx = jointIdx * self.featuresPerJoint;
			if(features.size > (baseIdx + 8), {
				velMag = (features.at(baseIdx + 3).squared +
				         features.at(baseIdx + 4).squared +
				         features.at(baseIdx + 5).squared).sqrt;

				accMag = (features.at(baseIdx + 6).squared +
				         features.at(baseIdx + 7).squared +
				         features.at(baseIdx + 8).squared).sqrt;

				jointMotion = velMag + (accMag * 0.3);

				if(jointMotion > 0.005, {
					totalMotion = totalMotion + jointMotion;
					activeJoints = activeJoints + 1;
				});
			});
		};

		if(activeJoints > 0, {
			totalMotion / activeJoints
		}, {
			0.0
		})
	},

	// Detect gesture start
	detectGestureStart: { |self, motionMagnitude, currentTime|
		var motionAboveThreshold, significantMotionIncrease;

		motionAboveThreshold = motionMagnitude > self.motionThreshold;
		significantMotionIncrease = motionMagnitude > (self.motionBaseline * 1.5);

		if(motionAboveThreshold || significantMotionIncrease, {
			self.isInGesture = true;
			self.gestureStartTime = currentTime;
			self.gestureFrames = List.new;
			self.totalGestureMotion = 0.0;
			self.peakVelocityInGesture = motionMagnitude;
			self.consecutiveLowMotionFrames = 0;
			self.gestureMotionProfile = List.new;

			if(self.debugMode, {
				("GESTURE START - motion: " ++ motionMagnitude.round(0.005)).postln;
			});

			true
		}, {
			false
		})
	},

	// Detect gesture end
	detectGestureEnd: { |self, motionMagnitude, currentTime|
		var timeSinceStart, motionDroppedSignificantly, motionBelowBaseline;
		var sufficientGestureLength, timeSinceLastMotion;
		var avgMotionPerFrame, adaptiveMaxPause, shouldEndGesture, endReason;

		timeSinceStart = currentTime - self.gestureStartTime;
		motionDroppedSignificantly = motionMagnitude < (self.peakVelocityInGesture * 0.3);
		motionBelowBaseline = motionMagnitude < (self.motionBaseline * 1.5);
		sufficientGestureLength = self.gestureFrames.size >= self.minGestureLength;
		timeSinceLastMotion = currentTime - self.lastMotionTime;

		avgMotionPerFrame = if(self.gestureFrames.size > 0, {
			self.totalGestureMotion / self.gestureFrames.size
		}, {
			0.0
		});
		adaptiveMaxPause = self.maxGesturePause * (1.0 + (avgMotionPerFrame * 2.0));

		if(motionBelowBaseline, {
			self.consecutiveLowMotionFrames = self.consecutiveLowMotionFrames + 1;
		}, {
			self.consecutiveLowMotionFrames = 0;
		});

		shouldEndGesture = false;
		endReason = "";

		if(motionDroppedSignificantly && (self.consecutiveLowMotionFrames >= 3), {
			shouldEndGesture = true;
			endReason = "velocity_decay";
		});

		if(timeSinceLastMotion > (adaptiveMaxPause * 0.7), {
			shouldEndGesture = true;
			endReason = "motion_pause";
		});

		if(self.gestureFrames.size > self.maxGestureLength, {
			shouldEndGesture = true;
			endReason = "max_length";
		});

		if(shouldEndGesture && sufficientGestureLength, {
			if(self.debugMode, {
				("GESTURE END - reason: " ++ endReason).postln;
			});
			true
		}, {
			if(shouldEndGesture && sufficientGestureLength.not, {
				if(self.debugMode, {
					("GESTURE ABANDONED - too short").postln;
				});
				self.isInGesture = false;
			});
			false
		})
	},

	// MODIFIED: DTW using effective joints
	dtwFullBody: { |self, seq1, seq2, gestureName|
		var n, m, distance, step, samplePoints, i1, i2;

		n = seq1.size;
		m = seq2.size;
		distance = 0.0;
		samplePoints = 0;

		if((max(n, m) / min(n, m).max(1)) > 4.0, {
			(self.recognitionThreshold * 3);
		}, {
			step = max(1, min(n, m) / 15).asInteger;

			0.to(14).do { |i|
				i1 = min(n-1, (i * n / 14.0).asInteger);
				i2 = min(m-1, (i * m / 14.0).asInteger);
				if((i1 < n) && (i2 < m), {
					distance = distance + self.euclideanDistanceFullBody(seq1.at(i1), seq2.at(i2), gestureName);
					samplePoints = samplePoints + 1;
				});
			};

			if(samplePoints > 0, {
				distance / samplePoints
			}, {
				self.recognitionThreshold * 2
			})
		});
	},

	// MODIFIED: Distance calculation for effective joints
	euclideanDistanceFullBody: { |self, a, b, gestureName|
		var distance, baseIdx, velDiff, accDiff, rotDiff, jerkDiff;
		var jointWeight, totalWeight, gestureWeights;

		distance = 0.0;
		totalWeight = 0.0;

		gestureWeights = if(self.adaptiveWeightingEnabled && gestureName.notNil, {
			self.getGestureWeights(gestureName);
		}, {
			Array.fill(self.effectiveNumJoints, 1.0);
		});

		self.effectiveNumJoints.do { |jointIdx|
			baseIdx = jointIdx * self.featuresPerJoint;

			jointWeight = gestureWeights.at(jointIdx);

			if((a.size > (baseIdx + 11)) && (b.size > (baseIdx + 11)), {
				rotDiff = (a.at(baseIdx + 0) - b.at(baseIdx + 0)).squared +
				         (a.at(baseIdx + 1) - b.at(baseIdx + 1)).squared +
				         (a.at(baseIdx + 2) - b.at(baseIdx + 2)).squared;

				velDiff = (a.at(baseIdx + 3) - b.at(baseIdx + 3)).squared +
				         (a.at(baseIdx + 4) - b.at(baseIdx + 4)).squared +
				         (a.at(baseIdx + 5) - b.at(baseIdx + 5)).squared;

				accDiff = (a.at(baseIdx + 6) - b.at(baseIdx + 6)).squared +
				         (a.at(baseIdx + 7) - b.at(baseIdx + 7)).squared +
				         (a.at(baseIdx + 8) - b.at(baseIdx + 8)).squared;

				jerkDiff = (a.at(baseIdx + 9) - b.at(baseIdx + 9)).squared +
				          (a.at(baseIdx + 10) - b.at(baseIdx + 10)).squared +
				          (a.at(baseIdx + 11) - b.at(baseIdx + 11)).squared;

				distance = distance + (jointWeight * (
					(rotDiff * 0.1) +
					(velDiff * 1.2) +
					(accDiff * 0.8) +
					(jerkDiff * 0.2)
				));

				totalWeight = totalWeight + jointWeight;
			});
		};

		if(totalWeight > 0, {
			(distance / totalWeight).sqrt
		}, {
			self.recognitionThreshold * 2
		})
	},

	// MODIFIED: Feature extraction for subset or all joints
	featureExtractor: { |self|
		var allFeatures, rot, rotVel, rotAcc, rotJerk, startIdx, velData, lastVel;
		var jointIdx, originalJointIdx;

		allFeatures = [];
		rot = self.lastOSCData.at(\rot);
		rotVel = self.lastOSCData.at(\rotVel);
		rotAcc = self.lastOSCData.at(\rotAcc);
		rotJerk = self.lastOSCData.at(\rotJerk);

		self.effectiveNumJoints.do { |ji|
			jointIdx = ji;

			// Map to original joint index if using subset
			originalJointIdx = if(self.useJointSubset, {
				self.jointSubsetMap.at(jointIdx);
			}, {
				jointIdx
			});

			startIdx = originalJointIdx * 3;

			// Get rotation (3 values)
			if(rot.size > (startIdx + 2), {
				allFeatures = allFeatures ++ rot.copyRange(startIdx, startIdx + 2);
			}, {
				allFeatures = allFeatures ++ [0, 0, 0];
			});

			// Get velocity (3 values) with smoothing
			if(rotVel.size > (startIdx + 2), {
				velData = rotVel.copyRange(startIdx, startIdx + 2);
				if(self.lastFeatures.notNil && (self.lastFeatures.size > (allFeatures.size + 2)), {
					lastVel = self.lastFeatures.copyRange(allFeatures.size, allFeatures.size + 2);
					velData = velData.collect { |v, i|
						(v * (1.0 - self.gestureVelocitySmoothing)) +
						(lastVel.at(i) * self.gestureVelocitySmoothing)
					};
				});
				allFeatures = allFeatures ++ velData;
			}, {
				allFeatures = allFeatures ++ [0, 0, 0];
			});

			// Get acceleration (3 values)
			if(rotAcc.size > (startIdx + 2), {
				allFeatures = allFeatures ++ rotAcc.copyRange(startIdx, startIdx + 2);
			}, {
				allFeatures = allFeatures ++ [0, 0, 0];
			});

			// Get jerk (3 values)
			if(rotJerk.size > (startIdx + 2), {
				allFeatures = allFeatures ++ rotJerk.copyRange(startIdx, startIdx + 2);
			}, {
				allFeatures = allFeatures ++ [0, 0, 0];
			});
		};

		self.lastFeatures = allFeatures;
		allFeatures
	},

	// Recognition using effective joints
	recognizeCompleteGestureFullBody: { |self|
		var candidates, distance, gestureName, confidence, bestMatch, currentTime;
		var gestureSequence, bestDistance, allConfidences, minMotionRequired;
		var testTemplate, allDistances, minDistance, maxDistance, normalizedDistance;
		var totalConf, totalMotion, avgMotionPerFrame;

		currentTime = SystemClock.seconds;

		if(self.gestureLibrary.size == 0, {
			nil;
		}, {
			gestureSequence = self.gestureFrames.asArray;

			if(gestureSequence.size < self.minGestureLength, {
				nil;
			}, {
				totalMotion = gestureSequence.collect { |frame|
					self.calculateMotionMagnitude(frame)
				}.sum;
				avgMotionPerFrame = totalMotion / gestureSequence.size;
				minMotionRequired = 0.008;

				if(avgMotionPerFrame < minMotionRequired, {
					if(self.debugMode, {
						("Gesture discarded - insufficient motion: " ++ avgMotionPerFrame.round(0.003)).postln;
					});
					nil;
				}, {
					candidates = [];
					allConfidences = Dictionary.new;
					allDistances = Dictionary.new;

					self.gestureLibrary.keysValuesDo { |name, templates|
						bestDistance = inf;

						testTemplate = if(self.useTemplateAveraging, {
							var avgTemplate = self.averagedTemplates.at(name);
							if(avgTemplate.isNil, {
								avgTemplate = self.averageTemplates(templates);
								self.averagedTemplates.put(name, avgTemplate);
							});
							[avgTemplate];
						}, {
							templates
						});

						testTemplate.do { |template, templateIdx|
							if(template.size >= self.minGestureLength, {
								try {
									distance = self.dtwFullBody(gestureSequence, template, name);
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

						allDistances.put(name, bestDistance);

						if(bestDistance < self.recognitionThreshold, {
							candidates = candidates ++ [[name, bestDistance]];
						});
					};

					if(allDistances.size > 0, {
						minDistance = allDistances.values.minItem;
						maxDistance = allDistances.values.maxItem;

						allDistances.keysValuesDo { |name, dist|
							var conf;

							if(dist >= self.recognitionThreshold, {
								conf = max(0, 0.1 - ((dist - self.recognitionThreshold) / self.recognitionThreshold));
								conf = conf.max(0);
							}, {
								if(minDistance == maxDistance, {
									conf = 0.5;
								}, {
									normalizedDistance = (dist - minDistance) / (maxDistance - minDistance);
									conf = exp(normalizedDistance.neg * 3) * (1 - (dist / self.recognitionThreshold));
									conf = conf.clip(0, 1);
								});
							});

							allConfidences.put(name, conf);
						};

						totalConf = allConfidences.values.sum;
						if(totalConf > 0, {
							allConfidences.keysValuesDo { |name, conf|
								allConfidences.put(name, conf / totalConf);
							};
						});
					});

					if(allConfidences.size > 0, {
						("--- Gesture Analysis ---").postln;
						allDistances.keysValuesDo { |name, dist|
							var conf = allConfidences.at(name);
							var status = if(dist < self.recognitionThreshold, {"✓"}, {"✗"});
							("  " ++ status ++ " " ++ name ++ ": " ++
							 (conf * 100).round(0.1) ++ "% (dist: " ++
							 dist.round(0.01) ++ ")").postln;
						};
					});

					if(candidates.size > 0, {
						candidates = candidates.sort { |a, b| a.at(1) < b.at(1) };
						bestMatch = candidates.at(0);
						gestureName = bestMatch.at(0);
						distance = bestMatch.at(1);
						confidence = allConfidences.at(gestureName);

						if((confidence > self.confidenceThreshold) && ((currentTime - self.lastDetectionTime) > self.detectionCooldown), {
							self.lastDetectionTime = currentTime;
							(">>> DETECTED: " ++ gestureName ++ " (confidence: " ++ (confidence * 100).round(0.1) ++ "%) <<<").postln;

							if(self.gestureCallbacks.at(gestureName).notNil, {
								try {
									self.gestureCallbacks.at(gestureName).(confidence);
								} { |error|
									("Callback error: " ++ error.errorString).postln;
								};
							});
						}, {
							if(self.debugMode, {
								if(confidence <= self.confidenceThreshold, {
									("Best match below confidence threshold: " ++ gestureName ++ " at " ++ (confidence * 100).round(0.1) ++ "%").postln;
								}, {
									("Cooldown active - waiting " ++ (self.detectionCooldown - (currentTime - self.lastDetectionTime)).round(0.1) ++ "s").postln;
								});
							});
						});
					}, {
						if(self.debugMode, {
							("No gestures matched (all distances above threshold " ++ self.recognitionThreshold ++ ")").postln;
						});
					});
				});
			});
		});
	},

	// MODIFIED: Setup OSC receiver to track actual data size
	setupOSCReceiver: { |self, port|
		if(port.notNil, {
			self.oscPort = port;
		});

		thisProcess.openUDPPort(self.oscPort);

		OSCdef(\mocapRot).free;
		OSCdef(\mocapRotVel).free;
		OSCdef(\mocapRotAcc).free;
		OSCdef(\mocapRotJerk).free;

		OSCdef(\mocapRot, { |msg|
			var values;
			values = msg.copyRange(1, msg.size-1);
			self.lastOSCData.put(\rot, values);
			self.actualDataSize.put(\rot, values.size);
			self.processFrame();
		}, ~oscRotationAddress);

		OSCdef(\mocapRotVel, { |msg|
			var values;
			values = msg.copyRange(1, msg.size-1);
			self.lastOSCData.put(\rotVel, values);
			self.actualDataSize.put(\rotVel, values.size);
		}, ~oscVelocityAddress);

		OSCdef(\mocapRotAcc, { |msg|
			var values;
			values = msg.copyRange(1, msg.size-1);
			self.lastOSCData.put(\rotAcc, values);
			self.actualDataSize.put(\rotAcc, values.size);
		}, ~oscAccelerationAddress);

		OSCdef(\mocapRotJerk, { |msg|
			var values;
			values = msg.copyRange(1, msg.size-1);
			self.lastOSCData.put(\rotJerk, values);
			self.actualDataSize.put(\rotJerk, values.size);
		}, ~oscJerkAddress);

		("OSC setup on port " ++ self.oscPort ++ " listening to:").postln;
		("  Rotation: " ++ ~oscRotationAddress).postln;
		("  Velocity: " ++ ~oscVelocityAddress).postln;
		("  Acceleration: " ++ ~oscAccelerationAddress).postln;
		("  Jerk: " ++ ~oscJerkAddress).postln;
	},

	// Continuous frame processing
	processFrame: { |self|
		var features, currentTime, motionMagnitude;

		self.frameSkipCounter = self.frameSkipCounter + 1;

		if(self.frameSkipCounter % self.frameSkipRate == 0, {
			currentTime = SystemClock.seconds;
			features = self.featureExtractor();

			motionMagnitude = self.calculateMotionMagnitude(features);
			self.updateMotionBaseline(motionMagnitude);
			self.currentVelocityMagnitude = (motionMagnitude * 0.3) + (self.currentVelocityMagnitude * 0.7);

			self.gestureBuffer = self.gestureBuffer.copyRange(1, self.gestureBuffer.size-1) ++ [features];

			if(self.isRecording, {
				self.currentGesture.add(features);
				self.recordingFrameCount = self.recordingFrameCount + 1;

				if(self.debugMode && ((self.recordingFrameCount % 20) == 0), {
					("Recording frame " ++ self.currentGesture.size).postln;
				});

				if(self.currentGesture.size > self.maxGestureLength, {
					("Gesture recording too long, stopping").postln;
					self.isRecording = false;
				});
			});

			if(self.isRecognizing && self.isRecording.not, {

				if(motionMagnitude > self.motionThreshold, {
					self.lastMotionTime = currentTime;
				});

				if(self.isInGesture.not, {
					self.detectGestureStart(motionMagnitude, currentTime);
				}, {
					self.gestureFrames.add(features);
					self.totalGestureMotion = self.totalGestureMotion + motionMagnitude;
					self.peakVelocityInGesture = max(self.peakVelocityInGesture, motionMagnitude);
					self.gestureMotionProfile.add(motionMagnitude);

					if(self.detectGestureEnd(motionMagnitude, currentTime), {
						self.recognizeCompleteGestureFullBody();
						self.isInGesture = false;
					});
				});
			});
		});
	},

	// MODIFIED: Auto-detect and allow dynamic updates
	autoDetectJoints: { |self|
		var rotSize, detectedJoints, totalFeatures;

		rotSize = self.actualDataSize.at(\rot);
		if(rotSize == 0, {
			rotSize = self.lastOSCData.at(\rot).size;
		});

		detectedJoints = (rotSize / 3).asInteger;

		if(detectedJoints > 0, {
			self.numJoints = detectedJoints;

			// If no subset is active, update effective joints
			if(self.useJointSubset.not, {
				self.effectiveNumJoints = detectedJoints;
				totalFeatures = self.effectiveNumJoints * self.featuresPerJoint;
				self.gestureBuffer = Array.fill(100, { Array.fill(totalFeatures, 0) });
			});

			("Auto-detected " ++ detectedJoints ++ " joints from " ++ rotSize ++ " rotation values").postln;
			if(self.useJointSubset, {
				("Using subset of " ++ self.effectiveNumJoints ++ " joints: " ++ self.jointSubset).postln;
			}, {
				("Using all " ++ self.effectiveNumJoints ++ " joints").postln;
			});
		}, {
			"No OSC data received yet for auto-detection".postln;
		});
	},

	// Start recording
	startRecording: { |self, gestureName|
		if(gestureName.isNil, {
			"Error: Gesture name required".postln;
		}, {
			self.currentGesture = List.new;
			self.recordingFrameCount = 0;
			self.isRecording = true;
			("Recording gesture: " ++ gestureName).postln;
			if(self.useJointSubset, {
				("Using joint subset: " ++ self.jointSubset).postln;
			});
		});
	},

	// Stop recording
	stopRecording: { |self, gestureName|
		var gestureToSave, templates, avgMotion, totalMotion, existingGestures;

		if(self.isRecording && (self.currentGesture.size >= self.minGestureLength), {
			gestureToSave = if(self.currentGesture.size > self.maxGestureLength, {
				self.currentGesture.asArray.copyRange(0, self.maxGestureLength-1);
			}, {
				self.currentGesture.asArray;
			});

			totalMotion = gestureToSave.collect { |frame| self.calculateMotionMagnitude(frame) }.sum;
			avgMotion = totalMotion / gestureToSave.size;

			existingGestures = self.gestureLibrary.at(gestureName);
			if(existingGestures.isNil, {
				self.gestureLibrary.put(gestureName, []);
				existingGestures = [];
				("Creating new gesture: " ++ gestureName).postln;
			}, {
				("Adding to existing gesture: " ++ gestureName).postln;
			});

			templates = existingGestures ++ [gestureToSave];

			if(templates.size > self.maxTemplatesPerGesture, {
				templates = templates.copyRange(templates.size - self.maxTemplatesPerGesture, templates.size-1);
				("Keeping only last " ++ self.maxTemplatesPerGesture ++ " templates").postln;
			});

			self.gestureLibrary.put(gestureName, templates);

			if(self.useTemplateAveraging, {
				self.averagedTemplates.removeAt(gestureName);
			});

			if(self.adaptiveWeightingEnabled, {
				self.learnGestureWeights(gestureName, gestureToSave);
			});

			self.isRecording = false;

			("Gesture '" ++ gestureName ++ "' saved - template " ++ templates.size ++ "/" ++ self.maxTemplatesPerGesture ++ " with " ++ gestureToSave.size ++ " frames").postln;
			("  Total motion: " ++ totalMotion.round(0.01) ++ ", Avg motion/frame: " ++ avgMotion.round(0.003)).postln;
			if(self.useJointSubset, {
				("  Used " ++ self.effectiveNumJoints ++ " joints from subset: " ++ self.jointSubset).postln;
			});

			if(avgMotion < 0.01, {
				("  WARNING: Very low motion detected - gesture may not be recognized well").postln;
			});

			if(gestureToSave.size < 15, {
				("  WARNING: Very short gesture - may be confused with noise").postln;
			});
		}, {
			if(self.isRecording, {
				("Gesture too short: " ++ self.currentGesture.size ++ " frames (min: " ++ self.minGestureLength ++ ")").postln;
			}, {
				"Not recording".postln;
			});
			self.isRecording = false;
		});

		nil
	},

	// Start recognition
	startRecognition: { |self|
		if(self.isRecognizing, {
			self.stopRecognition();
		});

		if(self.gestureLibrary.size == 0, {
			"No gestures in library".postln;
		}, {
			self.isRecognizing = true;
			self.recognitionTask = Routine({
				while { self.isRecognizing } {
					self.recognitionInterval.wait;
				};
			}).play;

			("Continuous recognition started with " ++ self.gestureLibrary.size ++ " gestures").postln;
			if(self.useJointSubset, {
				("Using " ++ self.effectiveNumJoints ++ " joints from subset: " ++ self.jointSubset).postln;
			}, {
				("Using all " ++ self.numJoints ++ " joints").postln;
			});
			("Recognition threshold: " ++ self.recognitionThreshold).postln;
			("Confidence threshold: " ++ (self.confidenceThreshold * 100).round(0.1) ++ "%").postln;
		});
	},

	// Other methods remain the same...
	stopRecognition: { |self|
		self.isRecognizing = false;
		if(self.recognitionTask.notNil, {
			self.recognitionTask.stop;
			self.recognitionTask = nil;
		});
		"Recognition stopped".postln;
	},

	setRecognitionThreshold: { |self, threshold|
		self.recognitionThreshold = threshold.clip(1.0, 50.0);
		("Recognition threshold set to: " ++ self.recognitionThreshold).postln;
	},

	setConfidenceThreshold: { |self, threshold|
		self.confidenceThreshold = threshold.clip(0.0, 1.0);
		("Confidence threshold set to: " ++ (self.confidenceThreshold * 100).round(0.1) ++ "%").postln;
	},

	setMotionSensitivity: { |self, level|
		var multipliers;
		level = level.clip(1, 5);
		multipliers = [0.3, 0.7, 1.0, 1.5, 2.0];
		self.motionThreshold = self.motionBaseline + (0.015 * multipliers[level-1]);
		("Motion sensitivity: " ++ level ++ "/5").postln;
	},

	setEndSensitivity: { |self, level|
		var pauseDurations, velocityDecays;
		level = level.clip(1, 5);
		pauseDurations = [0.3, 0.5, 0.7, 0.9, 1.2];
		velocityDecays = [0.2, 0.3, 0.4, 0.5, 0.6];
		self.maxGesturePause = pauseDurations[level-1];
		self.velocityDecayThreshold = velocityDecays[level-1];
		("End sensitivity: " ++ level ++ "/5").postln;
	},

	setAdaptiveWeighting: { |self, enabled|
		self.adaptiveWeightingEnabled = enabled;
		("Adaptive weighting: " ++ if(enabled, {"enabled"}, {"disabled"})).postln;
		if(enabled.not, {
			self.averagedTemplates.clear();
		});
	},

	setTemplateAveraging: { |self, enabled|
		self.useTemplateAveraging = enabled;
		("Template averaging: " ++ if(enabled, {"enabled"}, {"disabled"})).postln;
		if(enabled.not, {
			self.averagedTemplates.clear();
		});
	},

	// MODIFIED: Show joint subset info
	listGestures: { |self|
		if(self.gestureLibrary.size > 0, {
			"=== Gesture Library ===".postln;
			self.gestureLibrary.keysValuesDo { |name, templates|
				var frameCounts, gestureWeights;
				frameCounts = templates.collect(_.size);
				("  " ++ name ++ " (" ++ templates.size ++ " templates: " ++ frameCounts ++ " frames)").postln;

				if(self.adaptiveWeightingEnabled, {
					gestureWeights = self.gestureJointWeights.at(name);
					if(gestureWeights.notNil, {
						("    Important joints: ").post;
						gestureWeights.collect { |weight, idx|
							var originalIdx = if(self.useJointSubset, {
								self.jointSubsetMap.at(idx)
							}, {
								idx
							});
							[originalIdx, weight]
						}
							.sort { |a, b| a[1] > b[1] }
							.copyRange(0, 2.min(gestureWeights.size-1))
							.do { |pair|
								(pair[0].asString ++ " ").post;
							};
						("").postln;
					});
				});
			};
			if(self.useJointSubset, {
				("Joint subset active: " ++ self.jointSubset).postln;
			});
		}, {
			"Gesture library is empty".postln;
		});
	},

	clearLibrary: { |self|
		self.gestureLibrary = Dictionary.new;
		self.gestureCallbacks = Dictionary.new;
		self.gestureJointWeights = Dictionary.new;
		self.gestureMotionProfiles = Dictionary.new;
		self.crossGestureWeights = Dictionary.new;
		self.averagedTemplates = Dictionary.new;
		"Library cleared".postln;
	},

	setDebugMode: { |self, enabled|
		self.debugMode = enabled;
		("Debug mode: " ++ enabled).postln;
	},

	// MODIFIED: Show joint subset status
	getStatus: { |self|
		"=== Gesture Detector Status ===".postln;
		("Recording: " ++ self.isRecording).postln;
		("Recognizing: " ++ self.isRecognizing).postln;
		("Gestures in library: " ++ self.gestureLibrary.size).postln;
		("Total joints available: " ++ self.numJoints).postln;
		if(self.useJointSubset, {
			("Joint subset active: " ++ self.jointSubset).postln;
			("Effective joints: " ++ self.effectiveNumJoints).postln;
		}, {
			("Using all " ++ self.numJoints ++ " joints").postln;
		});
		("Current motion threshold: " ++ self.motionThreshold.round(0.003)).postln;
		("Motion baseline: " ++ self.motionBaseline.round(0.003)).postln;
		("Debug mode: " ++ self.debugMode).postln;
	},

	cleanup: { |self|
		self.stopRecognition();
		OSCdef(\mocapRot).free;
		OSCdef(\mocapRotVel).free;
		OSCdef(\mocapRotAcc).free;
		OSCdef(\mocapRotJerk).free;
		"Cleaned up".postln;
	},

	init: { |self, port|
		self.cleanup();
		self.setupOSCReceiver(port);
		self.autoDetectJoints();

		self.motionBaseline = 0.003;
		self.motionThreshold = 0.008;

		"Gesture detector initialized".postln;
		("OSC Port: " ++ self.oscPort).postln;
		if(self.useJointSubset, {
			("Using joint subset: " ++ self.jointSubset).postln;
		});
	}
);

// Initialize
~gestureDetector.init(~oscPort);

// ============================================
// OSC CONTROL FOR RECORDING
// ============================================
(
// Setup OSC control port (different from data input port)
thisProcess.openUDPPort(~controlPort);

// OSC to start recording
// Send: /record/start "gestureName"
OSCdef(\startRec, { |msg|
    var gestureName = msg[1].asString;
    if(gestureName.notNil and: { gestureName != "" }, {
        ("OSC: Starting recording - " ++ gestureName).postln;
        ~gestureDetector.startRecording(gestureName.asSymbol);
    }, {
        "OSC: ERROR - Gesture name required".postln;
    });
}, '/record/start', recvPort: ~controlPort);

// OSC to stop recording
// Send: /record/stop "gestureName"
OSCdef(\stopRec, { |msg|
    var gestureName = msg[1].asString;
    ("OSC: Stopping recording - " ++ gestureName).postln;
    ~gestureDetector.stopRecording(gestureName.asSymbol);
}, '/record/stop', recvPort: ~controlPort);

"".postln;
"=== OSC RECORDING CONTROL READY ===".postln;
("Control Port: " ++ ~controlPort).postln;
"Commands:".postln;
"  /record/start <name>  - Start recording".postln;
"  /record/stop <name>   - Stop recording".postln;
"".postln;
);
);

// ============================================
// USAGE EXAMPLES WITH JOINT SUBSETS
// ============================================
(
"".postln;
"=== JOINT SUBSET COMMANDS ===".postln;
"".postln;
"// Use only specific joints (e.g., right hand: joints 8,9,10):".postln;
"~gestureDetector.setJointSubset([8, 9, 10]);".postln;
"".postln;
"// Use a range of joints (e.g., upper body: joints 0-10):".postln;
"~gestureDetector.setJointRange(0, 10);".postln;
"".postln;
"// Use all joints again:".postln;
"~gestureDetector.setJointSubset(nil);".postln;
"".postln;
"// Common joint subsets:".postln;
"~gestureDetector.setJointSubset([15, 16, 17, 18, 19, 20, 21, 22]); // Both hands".postln;
"~gestureDetector.setJointSubset([0, 1, 2, 3, 4, 5, 6, 7]);        // Head and shoulders".postln;
"~gestureDetector.setJointSubset([8, 9, 10, 11, 12, 13, 14]);      // Arms".postln;
"".postln;
)
)

// Use only specified joints
~gestureDetector.setJointSubset([15, 16, 17, 18, 19, 20, 21, 22]);

// Use range
~gestureDetector.setJointRange(0, 10);

// Use specific joints
~gestureDetector.setJointSubset([8, 9, 10, 11, 12, 13, 14]);

// Return to using all joints
~gestureDetector.setJointSubset(nil);


// -------------------------EXAMPLE FLOW-------------------------------


//  Record gestures - weights are learned efficiently after recording
~gestureDetector.startRecording("wave");
// Perform waving gesture...
~gestureDetector.stopRecording("wave");

~gestureDetector.startRecording("punch");
~gestureDetector.stopRecording("punch");


//  Start recognition with optimized matching
~gestureDetector.startRecognition();

//  Monitor performance
~gestureDetector.getStatus();

//  Stop recognition
~gestureDetector.stopRecognition();

//  Enable debug to see optimization in action
~gestureDetector.setDebugMode(true);

//  Clear libraryy
~gestureDetector.clearLibrary();


// ============================================
// TEST: Send OSC from SuperCollider
// ============================================


// Test sending OSC messages to ourselves
~test = NetAddr("127.0.0.1", ~controlPort);

// Start recording a "shake" gesture
~test.sendMsg('/record/start', "shake");

// Stop recording (execute this after performing gesture)
~test.sendMsg('/record/stop', "shake");


~gestureDetector.startRecognition();


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

// -------------------------CONTROLS-------------------------------

// Make recognition more lenient for wave detection
~gestureDetector.setRecognitionThreshold(18);  //  10-20 recommended

// Lower confidence threshold
~gestureDetector.setConfidenceThreshold(0.35);  // 0.3-0.7 recommended


// Motion detection sensitivity (1=most sensitive):
~gestureDetector.setMotionSensitivity(2);  // 1-5
~gestureDetector.setEndSensitivity(3);     // 1-5

// Toggle features:

~gestureDetector.setAdaptiveWeighting(false);  // Disable adaptive weights
~gestureDetector.setTemplateAveraging(true);   // Enable template averaging

// View status:
~gestureDetector.getStatus();
~gestureDetector.listGestures();

// Check if template averaging is working
~gestureDetector.averagedTemplates.keys;  // Should show gesture names after first recognition

// If wave is still not detected, try disabling template averaging:
~gestureDetector.setTemplateAveraging(false);

// Or try enabling adaptive weighting to learn the difference:
~gestureDetector.setAdaptiveWeighting(true);


// For better speed invariance:
~gestureDetector.setMotionSensitivity(2);  // More sensitive to slow movements


