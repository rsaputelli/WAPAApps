from fnmatch import fnmatch

class JERouter:
    def __init__(self, cfg: dict):
        self.routes = cfg.get("routes", {})

    def credit_account_for_gl(self, gl: str, default: str | None = None) -> str | None:
        gl = (gl or "").strip()
        for pat, rule in self.routes.items():
            if fnmatch(gl, pat):
                return rule.get("credit_account")
        return default

def load_config(path: str) -> dict:
    import yaml, os
    with open(os.path.abspath(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
