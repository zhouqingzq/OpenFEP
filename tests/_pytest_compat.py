from __future__ import annotations


try:
    import pytest as _pytest
except ModuleNotFoundError:
    class _MarkProxy:
        def __getattr__(self, _name: str):
            def _decorator(target):
                return target

            return _decorator

    class _PytestStub:
        mark = _MarkProxy()

    pytest = _PytestStub()
else:
    pytest = _pytest
