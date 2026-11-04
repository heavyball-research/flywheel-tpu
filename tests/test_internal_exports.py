"""flywheel_tpu re-exports the internal names tpu-inference's backend imports,
each with a FutureWarning, and nothing else beyond its public API."""

import importlib

import pytest

import flywheel_tpu


@pytest.mark.parametrize(
    "name, module_name", sorted(flywheel_tpu.INTERNAL_EXPORTS.items())
)
def test_internal_export_warns_and_is_the_internal_object(name, module_name):
    with pytest.warns(FutureWarning, match=f"flywheel_tpu.{name} is internal"):
        value = getattr(flywheel_tpu, name)
    assert value is getattr(importlib.import_module(module_name), name)


def test_from_import_warns():
    with pytest.warns(FutureWarning, match="write_kv_cache_pages is internal"):
        # pylint: disable-next=import-outside-toplevel
        from flywheel_tpu import write_kv_cache_pages
    assert callable(write_kv_cache_pages)


def test_internal_exports_stay_out_of_all():
    assert not set(flywheel_tpu.INTERNAL_EXPORTS) & set(flywheel_tpu.__all__)


def test_unknown_name_raises():
    with pytest.raises(AttributeError, match="no attribute 'not_a_name'"):
        getattr(flywheel_tpu, "not_a_name")
