# mmpose
import warnings
from typing import Dict, List, Optional, Sequence, Union
import time

import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.infer.infer import ModelType
from mmengine.structures import InstanceData
from rich.progress import track

from mmpose.apis.inferencers.base_mmpose_inferencer import BaseMMPoseInferencer
from mmpose.apis.inferencers.hand3d_inferencer import Hand3DInferencer
from mmpose.apis.inferencers.pose2d_inferencer import Pose2DInferencer
from mmpose.apis.inferencers.pose3d_inferencer import Pose3DInferencer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ConfigType = Union[Config, ConfigDict]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]


config = {
    "init_args": None,
    "call_args": None
          }

class MMPoseInferencer(BaseMMPoseInferencer):
    """MMPose Inferencer. It's a unified inferencer interface for pose
    estimation task, currently including: Pose2D. and it can be used to perform
    2D keypoint detection.

    Args:
        pose2d (str, optional): Pretrained 2D pose estimation algorithm.
            It's the path to the config file or the model name defined in
            metafile. For example, it could be:

            - model alias, e.g. ``'body'``,
            - config name, e.g. ``'simcc_res50_8xb64-210e_coco-256x192'``,
            - config path

            Defaults to ``None``.
        pose2d_weights (str, optional): Path to the custom checkpoint file of
            the selected pose2d model. If it is not specified and "pose2d" is
            a model name of metafile, the weights will be loaded from
            metafile. Defaults to None.
        device (str, optional): Device to run inference. If None, the
            available device will be automatically used. Defaults to None.
        scope (str, optional): The scope of the model. Defaults to "mmpose".
        det_model(str, optional): Config path or alias of detection model.
            Defaults to None.
        det_weights(str, optional): Path to the checkpoints of detection
            model. Defaults to None.
        det_cat_ids(int or list[int], optional): Category id for
            detection model. Defaults to None.
        output_heatmaps (bool, optional): Flag to visualize predicted
            heatmaps. If set to None, the default setting from the model
            config will be used. Default is None.
    """

    preprocess_kwargs: set = {
        'bbox_thr', 'nms_thr', 'bboxes', 'use_oks_tracking', 'tracking_thr',
        'disable_norm_pose_2d'
    }
    forward_kwargs: set = {
        'merge_results', 'disable_rebase_keypoint', 'pose_based_nms'
    }
    visualize_kwargs: set = {
        'return_vis', 'show', 'wait_time', 'draw_bbox', 'radius', 'thickness',
        'kpt_thr', 'vis_out_dir', 'skeleton_style', 'draw_heatmap',
        'black_background', 'num_instances'
    }
    postprocess_kwargs: set = {'pred_out_dir', 'return_datasample'}

    def __init__(self,
                 pose2d: Optional[str] = None,
                 pose2d_weights: Optional[str] = None,
                 pose3d: Optional[str] = None,
                 pose3d_weights: Optional[str] = None,
                 device: Optional[str] = None,
                 scope: str = 'mmpose',
                 det_model: Optional[Union[ModelType, str]] = None,
                 det_weights: Optional[str] = None,
                 det_cat_ids: Optional[Union[int, List]] = None,
                 show_progress: bool = False) -> None:

        print("device ", device)

        self.visualizer = None
        self.show_progress = show_progress
        if pose3d is not None:
            if 'hand3d' in pose3d:
                self.inferencer = Hand3DInferencer(pose3d, pose3d_weights,
                                                   device, scope, det_model,
                                                   det_weights, det_cat_ids,
                                                   show_progress)
            else:
                self.inferencer = Pose3DInferencer(pose3d, pose3d_weights,
                                                   pose2d, pose2d_weights,
                                                   device, scope, det_model,
                                                   det_weights, det_cat_ids,
                                                   show_progress)
        elif pose2d is not None:
            self.inferencer = Pose2DInferencer(pose2d, pose2d_weights, device,
                                               scope, det_model, det_weights,
                                               det_cat_ids, show_progress)
        else:
            raise ValueError('Either 2d or 3d pose estimation algorithm '
                             'should be provided.')
            
    """
    by Daniel
    """
    def setVisualizer(self, visualizer):
        self.visualizer = visualizer
        
    """
    TODO by Daniel: set osc sender
    """

    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
            List[str or np.ndarray]: List of original inputs in the batch
        """
        
        for data in self.inferencer.preprocess(inputs, batch_size, **kwargs):
            yield data

    @torch.no_grad()
    def forward(self, inputs: InputType, **forward_kwargs) -> PredType:
        """Forward the inputs to the model.

        Args:
            inputs (InputsType): The inputs to be forwarded.

        Returns:
            Dict: The prediction results. Possibly with keys "pose2d".
        """
        
        #print("forward")
        
        return self.inferencer.forward(inputs, **forward_kwargs)

    def __call__(
        self,
        inputs: InputsType,
        return_datasamples: bool = False,
        batch_size: int = 1,
        out_dir: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                :obj:`BaseDataElement`. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.
            out_dir (str, optional): directory to save visualization
                results and predictions. Will be overoden if vis_out_dir or
                pred_out_dir are given. Defaults to None
            **kwargs: Key words arguments passed to :meth:`preprocess`,
                :meth:`forward`, :meth:`visualize` and :meth:`postprocess`.
                Each key in kwargs should be in the corresponding set of
                ``preprocess_kwargs``, ``forward_kwargs``,
                ``visualize_kwargs`` and ``postprocess_kwargs``.

        Returns:
            dict: Inference and visualization results.
        """

        if out_dir is not None:
            if 'vis_out_dir' not in kwargs:
                kwargs['vis_out_dir'] = f'{out_dir}/visualizations'
            if 'pred_out_dir' not in kwargs:
                kwargs['pred_out_dir'] = f'{out_dir}/predictions'

        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in set.union(self.inferencer.preprocess_kwargs,
                                self.inferencer.forward_kwargs,
                                self.inferencer.visualize_kwargs,
                                self.inferencer.postprocess_kwargs)
        }
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)

        self.inferencer.update_model_visualizer_settings(**kwargs)

        # preprocessing
        if isinstance(inputs, str) and inputs.startswith('webcam'):
            inputs = self.inferencer._get_webcam_inputs(inputs)
            batch_size = 1
            if not visualize_kwargs.get('show', False):
                warnings.warn('The display mode is closed when using webcam '
                              'input. It will be turned on automatically.')
            visualize_kwargs['show'] = True
        else:
            inputs = self.inferencer._inputs_to_list(inputs)
        self._video_input = self.inferencer._video_input
        if self._video_input:
            self.video_info = self.inferencer.video_info
            
        inputs = self.preprocess(
            inputs, batch_size=batch_size, **preprocess_kwargs)

        # forward
        if 'bbox_thr' in self.inferencer.forward_kwargs:
            forward_kwargs['bbox_thr'] = preprocess_kwargs.get('bbox_thr', -1)

        preds = []

        for proc_inputs, ori_inputs in (track(inputs, description='Inference')
                                        if self.show_progress else inputs):
            
            #print("inputs_ ", inputs)
            #print("proc_inputs ", proc_inputs)
            #print("ori_inputs ", ori_inputs)

            preds = self.forward(proc_inputs, **forward_kwargs)
            
            #print("pose2d_results ", self.inferencer._buffer['pose2d_results']);

            
            #print("self.inferencer.pose2d_model.visualizer.skeleton ", self.inferencer.pose2d_model.visualizer.skeleton)
            

            #print("ori_inputs t ", type(ori_inputs))
            #print("preds t ", type(preds))
            #print("preds l ", len(preds))

            # ori_input 0 is the input image
            #print("ori_input 0 s ", ori_inputs[0].shape)

            # preds 0 is mmpose.structures.pose_data_sample.PoseDataSample
            # pred_instances within PoseDataSample is mmengine.structures.instance_data.InstanceData
            #print("pred_instances t ", type(preds[0].pred_instances))
            #print("pred_instances l ", len(preds[0].pred_instances))

            """
            pose 2d instance data fields:
                keypoint_scores: joint_count
                bbox_scores: box_count
                keypoints_visible: joint_count
                bboxes: box_count x 4
                keypoints: joint_count x 2
                
            pose 3d instance data fields:
                keypoints: joint_count x 3
                keypoint_scores: joint_count
            """
            
            """
            for p, pred in enumerate(preds):
                print("p ", p)
                
                for i, instance_data in enumerate(pred.pred_instances):
                    print("i ", i, " t ", type(instance_data))
                    print(instance_data)
            """

            """
            for i, instance_data in enumerate(preds[0].pred_instances):
                print("i ", i, " t ", type(instance_data))
                print(instance_data)
            """

            """
            for i, ori in enumerate(ori_inputs):
                print("i ", i, " ori t ", type(ori))
            for i, pred in enumerate(preds):
                print("i ", i, " pred t ", type(pred))
            """
            
            # collect all pose estimation information
            
            input_image = ori_inputs[0]
            
            #print("self.inferencer t ", type(self.inferencer))
            
            if "Pose3DInferencer" in str(type(self.inferencer)):
                
                pose2d_results = self.inferencer._buffer['pose2d_results']
                pose3d_results = preds[0]
                
                #print("is Pose3DInferencer")
            elif "Pose2DInference" in str(type(self.inferencer)):
                
                pose2d_results = preds[0]
                pose3d_results = None
                
                #print("is Pose2DInferencer")
            else:
                
                pose2d_results = None
                pose3d_results = None
                
                print("unknown Pose Inferencer")
            
            #print("pose2d_results t ", type(pose2d_results))
            #print("pose3d_results t ", type(pose3d_results))
            
            #pose2d_results = self.inferencer._buffer['pose2d_results']
            #pose3d_results = preds[0]
            
            """
            pose3d_results = self.postprocess(
                preds,
                None,
                return_datasamples=return_datasamples,
                **postprocess_kwargs)
            """
            
            results = {}
            results["image"] = input_image
            results["pose2d_results"] = pose2d_results
            results["pose3d_results"] = pose3d_results
            
            #print("pose2d_results t ", type(pose2d_results))
            #print("preds t ", type(preds[0]))
            #print("pose3d_results t ", type(pose3d_results))

            
            yield results

        if self._video_input:
            self._finalize_video_processing(
                postprocess_kwargs.get('pred_out_dir', ''))



class MotionModel():
    
    def __init__(self, config):
        super().__init__()
        
        self.init_args = config["init_args"]
        self.call_args = config["call_args"]

        self.model = MMPoseInferencer(**self.init_args)
        #self.generator = self.model(**self.call_args)
        
        self.generator = None

    def start(self):
        
        self.generator = self.model(**self.call_args)
        
    def update(self):
        
        if self.generator is None:
            return
        
        self.results = next(self.generator)
        
        #print("self.results ", self.results)
        #print(self.results)

        