import os
import glob
import shutil
import pickle
from datetime import datetime

import torch
import torch.fx as fx

from torch.fx.graph_module import GraphModule
from torch.fx.node import Argument, Target, Node, map_arg
from torch.fx._compatibility import compatibility
from torch.fx.immutable_collections import immutable_dict, immutable_list

from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from torch.hub import tqdm

from typing import Iterable


class CacheDataLoader:
    def __init__(self, path: str, ext: str = 'pkl') -> None:

        paths = os.path.join(path, f'*.{ext}')
        
        self.ext = ext
        self.path = path
        self.paths = glob.glob(paths)
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Any:

        with open(self.paths[idx], 'rb') as f:
            data = pickle.load(f)

        return data
    
    def append(self, sample: Any) -> None:
        
        n = len(self)
        path = os.path.join(self.path, f'sample{n}.{self.ext}')
        with open(path, 'wb') as f:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.paths.append(path)

    def cleanup(self):
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            pass


class DataLoader:
    def __init__(self, data: Union[List, str]) -> None:
        
        if isinstance(data, str):
            data = CacheDataLoader(data)

        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]
    
    def add_sample(self, sample: Any) -> None:
        self.data.append(sample)

    def cleanup(self):
        if isinstance(self.data, CacheDataLoader):
            self.data.cleanup()

        del self.data


def reinit_dataloader(
    data: Iterable
) -> Iterable:
    
    if isinstance(data[0], dict):
        raise RuntimeError('Positional arguments are not supported.')

    if not isinstance(data[0], (tuple, list)):
        dataloader = DataLoader([]) # TODO: add path arg if cache_data is enabled
        for sample in data:
            dataloader.add_sample(sample)
        return (dataloader,)

    n_args = len(data[0])
    dataloader = []
    for i in range(n_args):
        data_ldr = DataLoader([]) # TODO: add path arg if cache_data is enabled
        for sample in data:
            data_ldr.add_sample(sample[i])
        dataloader.append(data_ldr)

    return dataloader


class DataInterpreter(fx.Interpreter):

    @compatibility(is_backward_compatible=True)
    def __init__(self, module: GraphModule, garbage_collect_values: bool = True, cache_data: bool = False, cache_path: str = 'interpreter_outputs') -> None:
        assert isinstance(module, GraphModule)
        self.module = module
        self.submodules = dict(self.module.named_modules())
        self.env : Dict[Node, Any] = {}
        self.name = "DataInterpreter"
        self.garbage_collect_values = garbage_collect_values
        self.extra_traceback = True
        self.device = next(iter(module.parameters())).device

        if self.garbage_collect_values:

            self.cache_data = cache_data

            if cache_data:
                self._cache_path = cache_path
                self.cache_path = os.path.join(cache_path, datetime.now().strftime('%y%m%d%H%M%S%f'))
                os.makedirs(cache_path, exist_ok=True)

            self.user_to_last_uses : Dict[Node, List[Node]] = {}
            self.user_count: Dict[Node, int] = {}
            self.named_nodes: Dict[str, Node] = {node.name: node for node in module.graph.nodes}

            def register_last_uses(n : Node, user : Node):
                self.user_to_last_uses.setdefault(user, []).append(n)
                self.user_count.setdefault(n, 0)
                self.user_count[n] += 1

            for node in reversed(self.module.graph.nodes):
                map_arg(node.args, lambda n: register_last_uses(n, node))
                map_arg(node.kwargs, lambda n: register_last_uses(n, node))

    @staticmethod
    def _size_aggregate(args: Argument):

        if isinstance(args, dict):
            args = tuple(args.values())

        if isinstance(args, (list, tuple)):

            for arg in args:
                n = DataInterpreter._size_aggregate(arg)
                if n is not None:
                    return n
            return None
        
        elif isinstance(args, slice):
            return DataInterpreter._size_aggregate(args.start) or DataInterpreter._size_aggregate(args.stop) or DataInterpreter._size_aggregate(args.step)
        elif isinstance(args, DataLoader):
            return len(args)
        else:
            return None

    @staticmethod
    def _data_size(args: Tuple[Argument, ...], kwargs: Dict[str, Any]) -> Union[None, int]:
        n = DataInterpreter._size_aggregate(list(args) + list(kwargs.values()))
        return n
    
    @staticmethod
    def _map_aggregate(a: Argument, i, device: torch.device) -> Argument:

        '''
        Based on torch.fx.node.map_aggregate
        '''

        if isinstance(a, tuple):
            t = tuple(DataInterpreter._map_aggregate(elem, i, device) for elem in a)
            return t if not hasattr(a, '_fields') else type(a)(*t)
        elif isinstance(a, list):
            return immutable_list(DataInterpreter._map_aggregate(elem, i, device) for elem in a)
        elif isinstance(a, dict):
            return immutable_dict((k, DataInterpreter._map_aggregate(v, i, device)) for k, v in a.items())
        elif isinstance(a, slice):
            return slice(DataInterpreter._map_aggregate(a.start, i, device), DataInterpreter._map_aggregate(a.stop, i, device), DataInterpreter._map_aggregate(a.step, i, device))
        elif isinstance(a, DataLoader):
            return DataInterpreter._map_aggregate(a[i], i, device)
        elif isinstance(a, torch.Tensor):
            return a.to(device)
        else:
            return a

    def _process_args(self, n: int, args: Tuple[Argument, ...]) -> Any:

        for i in range(n):
            sample = tuple(DataInterpreter._map_aggregate(arg, i, self.device) for arg in args)
            yield sample

    def _process_kwargs(self, n: int, kwargs: Dict[str, Any]) -> Any:

        for i in range(n):
            sample = {k: DataInterpreter._map_aggregate(arg, i, self.device) for k, arg in kwargs.items()}
            yield sample

    @compatibility(is_backward_compatible=True)
    def call_function(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], save_path: Union[None, str] = None) -> Any:
        assert not isinstance(target, str)

        n = self._data_size(args, kwargs)
        
        if n is None:
            return target(*args, **kwargs)
        
        args = self._process_args(n, args)
        kwargs = self._process_kwargs(n, kwargs)

        results = DataLoader(save_path if save_path else [])
        for arg, kwarg in zip(args, kwargs):
            result = target(*arg, **kwarg)
            results.add_sample(result)

        return results

    @compatibility(is_backward_compatible=True)
    def call_method(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], save_path: Union[None, str] = None) -> Any:
        
        assert isinstance(target, str)

        n = self._data_size(args, kwargs)

        if n is None:
            self_obj, *args_tail = args
            return getattr(self_obj, target)(*args_tail, **kwargs)

        args = self._process_args(n, args)
        kwargs = self._process_kwargs(n, kwargs)

        results = DataLoader(save_path if save_path else [])
        for arg, kwarg in zip(args, kwargs):
            self_obj, *arg = arg
            result = getattr(self_obj, target)(*arg, **kwarg)
            results.add_sample(result)

        return results

    @compatibility(is_backward_compatible=True)
    def call_module(self, target: 'Target', args: Tuple[Argument, ...], kwargs: Dict[str, Any], save_path: Union[None, str] = None) -> Any:
        assert isinstance(target, str)
        submod = self.fetch_attr(target)

        n = self._data_size(args, kwargs)

        if n is None:
            return submod(*args, **kwargs)

        args = self._process_args(n, args)
        kwargs = self._process_kwargs(n, kwargs)

        results = DataLoader(save_path if save_path else [])
        for arg, kwarg in zip(args, kwargs):
            result = submod(*arg, **kwarg)
            results.add_sample(result)

        return results
    
    @compatibility(is_backward_compatible=True)
    def placeholder(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any], save_path: Union[None, str] = None) -> Any:
        return super().placeholder(target, args, kwargs)

    @compatibility(is_backward_compatible=True)
    def get_attr(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any], save_path: Union[None, str] = None) -> Any:
        return super().get_attr(target, args, kwargs)
    
    @compatibility(is_backward_compatible=True)
    def output(self, target : 'Target', args : Tuple[Argument, ...], kwargs : Dict[str, Any], save_path: Union[None, str] = None) -> Any:
        return super().output(target, args, kwargs)
    
    @compatibility(is_backward_compatible=True)
    def run_node(self, n : Node) -> Any:

        with self._set_current_node(n):
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)

            if hasattr(self, 'cache_data') and self.cache_data:
                cache_path = os.path.join(self.cache_path, n.name)
                os.makedirs(cache_path, exist_ok=True)
            else:
                cache_path = None

            return getattr(self, n.op)(n.target, args, kwargs, cache_path)
    
    @compatibility(is_backward_compatible=True)
    def _run_node(self, node):

        try:
            self.env[node] = self.run_node(node)
        except Exception as e:
            if self.extra_traceback:
                msg = f"While executing {node.format_node()}"
                msg = f'{e.args[0]}\n\n{msg}' if e.args else str(msg)
                msg += f"\nOriginal traceback:\n{node.stack_trace}"
                e.args = (msg,) + e.args[1:]
                if isinstance(e, KeyError):
                    raise RuntimeError(*e.args) from e
            raise

        if self.garbage_collect_values:
            for to_delete in self.user_to_last_uses.get(node, []):
                self.user_count[to_delete] -= 1
                if self.user_count[to_delete] < 1:
                    
                    if isinstance(self.env[to_delete], DataLoader):
                        self.env[to_delete].cleanup()

                    del self.env[to_delete]


    @compatibility(is_backward_compatible=True)
    def initialize_env(self, *args, initial_env : Optional[Dict[Node, Any]] = None, enable_io_processing: bool = True) -> None:

        if len(args) != 1:
            raise RuntimeError('DataInterpreter expects one dataloder!')
        
        args = reinit_dataloader(args[0])
        self.env = initial_env if initial_env is not None else {}

        if enable_io_processing:
            args = self.module.graph.process_inputs(*args)

        self.args_iter: Iterator[Any] = iter(args)
    
    @compatibility(is_backward_compatible=True)
    def run(self, *args, initial_env: Optional[Dict[Node, Any]] = None, enable_io_processing: bool = True) -> Any:

        self.initialize_env(*args, initial_env, enable_io_processing)

        pbar = tqdm(total=len(self.module.graph.nodes),
                    desc=f"{self.name}: {str(list(self.module.graph.nodes)) if fx.config.verbose_progress else ''}",
                    initial=0, position=0, leave=True, disable=fx.config.disable_progress, delay=0)

        for node in self.module.graph.nodes:
            pbar.update(1)
            if node in self.env:
                continue

            self._run_node(node)

            if node.op == 'output':
                output_val = self.env[node]
                return self.module.graph.process_outputs(output_val) if enable_io_processing else output_val
            
    def add_nodes_to_track(self, nodes: Tuple[Node]) -> None:
        assert self.garbage_collect_values

        for node in nodes:
            node = self.named_nodes[node.name]
            self.user_count.setdefault(node, 0)
            self.user_count[node] += 1

    def del_nodes_to_track(self, nodes: Tuple[Node]) -> None:
        assert self.garbage_collect_values

        for node in nodes:
            node = self.named_nodes[node.name]

            if self.user_count[node] < 1:
                if node in self.env:
                    raise RuntimeError('Detected memory leak')
                continue

            self.user_count[node] -= 1
            if self.user_count[node] < 1:
                
                if isinstance(self.env[node], DataLoader):
                    self.env[node].cleanup()

                del self.env[node]

    def __del__(self):

        del self.env

        if hasattr(self, 'cache_data') and self.cache_data:

            try:
                shutil.rmtree(self.cache_path)
            except FileNotFoundError:
                pass
        
            if os.path.exists(self._cache_path) and len(os.listdir(self._cache_path)) == 0:
                os.rmdir(self._cache_path)
