import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_EMBEDDERS = ROOT / "experiments" / "shared" / "baseline_embedders.py"
PIPELINE = ROOT / "experiments" / "shared" / "pipeline.py"
V5_MAIN = ROOT / "experiments" / "v5_baselines" / "main.py"


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name)


def _function(node: ast.AST, name: str) -> ast.FunctionDef:
    body = getattr(node, "body", [])
    return next(item for item in body if isinstance(item, ast.FunctionDef) and item.name == name)


def _literal_assigned(module: ast.Module, name: str):
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not assigned")


class V5BaselineStaticContractTest(unittest.TestCase):
    def test_mindid_preprocesses_before_delta_filtering(self):
        module = _module(BASELINE_EMBEDDERS)
        forward = _function(_class(module, "MindIDEmbedder"), "forward")

        first_delta_call = next(
            i for i, node in enumerate(forward.body)
            if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "_delta_filter"
                   for call in ast.walk(node))
        )
        first_mean = next(
            i for i, node in enumerate(forward.body)
            if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "mean"
                   for call in ast.walk(node))
        )
        first_std = next(
            i for i, node in enumerate(forward.body)
            if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "std"
                   for call in ast.walk(node))
        )

        self.assertLess(first_mean, first_delta_call)
        self.assertLess(first_std, first_delta_call)

    def test_brainnet_default_embedding_width_matches_paper_figure(self):
        module = _module(BASELINE_EMBEDDERS)
        init = _function(_class(module, "BrainNetEmbedder"), "__init__")
        defaults = dict(zip(
            [arg.arg for arg in init.args.args[-len(init.args.defaults):]],
            init.args.defaults,
        ))

        self.assertEqual(ast.literal_eval(defaults["embed_dim"]), 32)

    def test_v5_brainnet_entry_declares_triplet_and_32d_embedding(self):
        models = _literal_assigned(_module(V5_MAIN), "BASELINE_MODELS")
        brainnet = next(model for model in models if model["embedder"] == "brainnet")

        self.assertEqual(brainnet["name"], "BrainNet_Triplet")
        self.assertEqual(brainnet["loss"], "triplet")
        self.assertEqual(brainnet["embed_dim"], 32)

    def test_shared_smoke_runner_accepts_and_uses_model_specs(self):
        module = _module(PIPELINE)
        run_smoke_test = _function(module, "run_smoke_test")
        arg_names = [arg.arg for arg in run_smoke_test.args.args]
        source_names = {node.id for node in ast.walk(run_smoke_test) if isinstance(node, ast.Name)}
        string_literals = {node.value for node in ast.walk(run_smoke_test)
                           if isinstance(node, ast.Constant) and isinstance(node.value, str)}

        self.assertIn("models", arg_names)
        self.assertIn("models", source_names)
        self.assertIn("embedder", string_literals)
        self.assertIn("embed_dim", string_literals)


if __name__ == "__main__":
    unittest.main()
