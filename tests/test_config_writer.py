"""配置快照须保留模块各自的说明和可加载的参数值。"""

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Common.ConfigWriter import save_config


class ConfigWriterChecks(unittest.TestCase):
    def test_repeated_parameter_names_keep_their_module_comments(self):
        template = """CropPointCloud:
  parameters:
    mode: none  # 裁剪策略
TemporalFilter:
  parameters:
    mode: review  # 时序筛选策略
"""
        data = yaml.safe_load(template)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "template.yaml"
            output = Path(folder) / "snapshot.yaml"
            source.write_text(template, encoding="utf-8")
            save_config(output, data, [source])
            content = output.read_text(encoding="utf-8")
            crop = next(line for line in content.splitlines() if "mode: none" in line)
            temporal = next(line for line in content.splitlines() if "mode: review" in line)
            self.assertIn("# 裁剪策略", crop)
            self.assertIn("# 时序筛选策略", temporal)
            self.assertEqual(yaml.safe_load(content), data)


if __name__ == "__main__":
    unittest.main()
