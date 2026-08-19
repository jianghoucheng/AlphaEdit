from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import nn


class Trace(AbstractContextManager):
    """Temporarily trace or edit the input/output of one PyTorch module."""

    def __init__(
        self,
        module: nn.Module,
        layer: str | None = None,
        *,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Callable[[Any, str | None], Any] | None = None,
        stop: bool = False,
    ) -> None:
        self.layer = layer
        traced_module = get_module(module, layer) if layer is not None else module

        def retain_hook(_module, inputs, output):
            if retain_input:
                value = inputs[0] if len(inputs) == 1 else inputs
                self.input = _copy_value(
                    value,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )
            if edit_output is not None:
                output = edit_output(output, self.layer)
            if retain_output:
                self.output = _copy_value(
                    output,
                    clone=clone,
                    detach=detach,
                    retain_grad=retain_grad,
                )
                if retain_grad:
                    output = _copy_value(
                        self.output,
                        clone=True,
                        detach=False,
                        retain_grad=False,
                    )
            if stop:
                raise StopForward
            return output

        self._hook = traced_module.register_forward_hook(retain_hook)
        self._stop = stop

    def __enter__(self) -> Trace:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return bool(
            self._stop and exc_type is not None and issubclass(exc_type, StopForward)
        )

    def close(self) -> None:
        self._hook.remove()


class TraceDict(OrderedDict, AbstractContextManager):
    """Trace several named modules for the duration of a context."""

    def __init__(
        self,
        module: nn.Module,
        layers: Iterable[str],
        *,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Callable[[Any, str | None], Any] | None = None,
        stop: bool = False,
    ) -> None:
        super().__init__()
        layer_names = list(dict.fromkeys(layers))
        self._stop = stop
        for index, layer in enumerate(layer_names):
            self[layer] = Trace(
                module,
                layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and index == len(layer_names) - 1,
            )

    def __enter__(self) -> TraceDict:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return bool(
            self._stop and exc_type is not None and issubclass(exc_type, StopForward)
        )

    def close(self) -> None:
        for trace in reversed(self.values()):
            trace.close()


class StopForward(Exception):
    """Signal that a traced forward pass has collected everything needed."""


def set_requires_grad(requires_grad: bool, *objects: Any) -> None:
    for value in objects:
        if isinstance(value, nn.Module):
            for parameter in value.parameters():
                parameter.requires_grad_(requires_grad)
        elif isinstance(value, (nn.Parameter, torch.Tensor)):
            value.requires_grad_(requires_grad)
        else:
            raise TypeError(f"Cannot set requires_grad on {type(value)!r}")


def get_module(model: nn.Module, name: str) -> nn.Module:
    try:
        return model.get_submodule(name)
    except AttributeError as error:
        raise LookupError(name) from error


def get_parameter(model: nn.Module, name: str) -> nn.Parameter:
    try:
        return model.get_parameter(name)
    except AttributeError as error:
        raise LookupError(name) from error


def _copy_value(
    value: Any,
    *,
    clone: bool,
    detach: bool,
    retain_grad: bool,
) -> Any:
    if isinstance(value, torch.Tensor):
        if retain_grad:
            if not value.requires_grad:
                value.requires_grad_(True)
            value.retain_grad()
        elif detach:
            value = value.detach()
        return value.clone() if clone else value
    if isinstance(value, dict):
        return {
            key: _copy_value(
                item,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _copy_value(
                item,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            _copy_value(
                item,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
            )
            for item in value
        ]
    raise TypeError(f"Cannot copy traced value of type {type(value)!r}")
