import subprocess
import tempfile
from pathlib import Path

from tools.subgraph.core import RegionError
from tools.subgraph.runner import repository_identity

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestSourceIdentity(TestCase):
    def setUp(self):
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.git("init", "-q")
        (self.root / "compiler.py").write_text("baseline\n")
        self.git("add", "compiler.py")
        self.git("commit", "-qm", "Test fixture baseline")
        self.baseline = self.git("rev-parse", "HEAD").strip()

    def git(self, *args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=Subgraph Test",
                "-c",
                "user.email=subgraph-test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            text=True,
        )

    @parametrize("state", ["untracked", "unstaged", "staged", "committed"])
    def test_tool_changes_preserve_compiler_identity(self, state):
        path = self.root / "tools/subgraph/tool.py"
        path.parent.mkdir(parents=True)
        path.write_text("original\n")
        if state == "unstaged":
            self.git("add", "tools/subgraph")
            self.git("commit", "-qm", "Test fixture tool")
            path.write_text("updated\n")
        elif state in ("staged", "committed"):
            self.git("add", "tools/subgraph")
            if state == "committed":
                self.git("commit", "-qm", "Test fixture tool")
        identity = repository_identity(self.root, self.baseline)
        self.assertEqual(identity["compiler_reference"], self.baseline)
        self.assertEqual(identity["commit"], self.git("rev-parse", "HEAD").strip())
        self.assertEqual(identity["tool_changes"], ["tools/subgraph/tool.py"])
        self.assertTrue(identity["compiler_matches_reference"])

    @parametrize("state", ["unstaged", "staged", "committed", "index_only"])
    def test_compiler_changes_rejected(self, state):
        path = self.root / "compiler.py"
        path.write_text("changed\n")
        if state != "unstaged":
            self.git("add", "compiler.py")
        if state == "committed":
            self.git("commit", "-qm", "Test fixture changed compiler")
        elif state == "index_only":
            path.write_text("baseline\n")
        with self.assertRaisesRegex(RegionError, "SOURCE_IDENTITY"):
            repository_identity(self.root, self.baseline)

    @parametrize("path", ["torch/new_operator.py", "tools/subgraph_extra/tool.py"])
    def test_untracked_source_rejected(self, path):
        destination = self.root / path
        destination.parent.mkdir(parents=True)
        destination.write_text("unexpected\n")
        with self.assertRaisesRegex(RegionError, "SOURCE_IDENTITY"):
            repository_identity(self.root, self.baseline)

    def test_clean_baseline(self):
        identity = repository_identity(self.root, self.baseline)
        self.assertEqual(identity["commit"], self.baseline)
        self.assertEqual(identity["tool_changes"], [])


instantiate_parametrized_tests(TestSourceIdentity)


if __name__ == "__main__":
    run_tests()
