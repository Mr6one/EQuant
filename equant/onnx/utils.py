import io
import onnxruntime as ort

import torch
from torch.onnx import TrainingMode, OperatorExportTypes
from typing import Any, Collection, Mapping, Optional, Sequence, Tuple, Type, Union


__all__  = [
    'optimize_model',
    'export'
]


def optimize_model(
    model_path: str, 
    output_model_path: str = None
) -> None:
    
    output_model_path = output_model_path or model_path
    
    session_options = ort.SessionOptions()
    session_options.optimized_model_filepath = output_model_path
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    _ = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])


def export(
    model: Union[torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction],
    args: Union[Tuple[Any, ...], torch.Tensor],
    f: Union[str, io.BytesIO],
    export_params: bool = True,
    verbose: bool = False,
    training: TrainingMode = TrainingMode.EVAL,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    operator_export_type: OperatorExportTypes = OperatorExportTypes.ONNX,
    opset_version: Optional[int] = None,
    do_constant_folding: bool = True,
    dynamic_axes: Optional[
        Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]
    ] = None,
    keep_initializers_as_inputs: Optional[bool] = None,
    custom_opsets: Optional[Mapping[str, int]] = None,
    export_modules_as_functions: Union[bool, Collection[Type[torch.nn.Module]]] = False,
    autograd_inlining: Optional[bool] = True,
    optimize: bool = False
) -> None:
    
    torch.onnx.export(
        model,
        args,
        f,
        export_params,
        verbose,
        training,
        input_names,
        output_names,
        operator_export_type=operator_export_type,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        dynamic_axes=dynamic_axes,
        keep_initializers_as_inputs=keep_initializers_as_inputs,
        custom_opsets=custom_opsets,
        export_modules_as_functions=export_modules_as_functions,
        autograd_inlining=autograd_inlining
    )

    if optimize:
        optimize_model(f)
