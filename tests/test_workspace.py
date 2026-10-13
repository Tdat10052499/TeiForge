"""Tests for the workspace sandbox."""

import os

import pytest

from teiforge.workspace import Workspace, WorkspaceError


def test_relative_paths_resolve_inside_root(tmp_path):
    space = Workspace(tmp_path)
    assert space.resolve("a/b.txt") == (tmp_path / "a" / "b.txt").resolve()
    assert space.relative(space.resolve("a/b.txt")) == os.path.join("a", "b.txt")


def test_absolute_path_inside_root_is_allowed(tmp_path):
    target = tmp_path / "inside.txt"
    assert Workspace(tmp_path).resolve(str(target)) == target.resolve()


@pytest.mark.parametrize("path", ["..", "../x", "/", "a/../../x"])
def test_escaping_paths_are_rejected(tmp_path, path):
    with pytest.raises(WorkspaceError, match="outside the workspace"):
        Workspace(tmp_path / "root").resolve(path)


def test_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    link = root / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    with pytest.raises(WorkspaceError):
        Workspace(root).resolve("link.txt")


def test_empty_path_is_rejected(tmp_path):
    with pytest.raises(WorkspaceError, match="No file path"):
        Workspace(tmp_path).resolve("   ")


def test_default_root_follows_env_then_cwd(monkeypatch, tmp_path, isolated_environment):
    space = Workspace()
    assert space.root == isolated_environment.resolve()
    other = tmp_path / "env-root"
    other.mkdir()
    monkeypatch.setenv("TEIFORGE_WORKSPACE", str(other))
    assert space.root == other.resolve()
    assert "Workspace(root=None)" == repr(space)


def test_relative_falls_back_to_absolute_for_outside_paths(tmp_path):
    space = Workspace(tmp_path / "root")
    assert space.relative(tmp_path) == str(tmp_path)
