"""统一批次目录；只修改顶部参数即可切换批次或独立试验。"""

import errno
import os
import shutil
from pathlib import Path


class OutputLayout:
    """仅推导结构目录；参数来自调用者读取的 YAML。"""

    def __init__(self, root_dir, settings):
        self.root_dir = Path(root_dir).resolve()

        def name(value):
            if (
                not isinstance(value, str)
                or not value
                or value in (".", "..")
                or Path(value).name != value
                or "\\" in value
            ):
                raise ValueError(f"目录名必须是单个名称：{value!r}")
            return value

        self.batch_name = name(settings["batch_name"])
        self.experiment_name = settings["experiment_name"]
        self.run_root = self.root_dir / "run" / self.batch_name
        self.active_root = (
            self.run_root / "experiments" / name(self.experiment_name)
            if self.experiment_name
            else self.run_root
        )
        self.work_dir = self.active_root / "work"
        self.result_dir = self.active_root / "result"
        self.scan_root = self.work_dir / "01_preprocess"
        self.scan_name = settings.get("scan_name", "")
        self.scan_dir = self.scan_root / name(self.scan_name) if self.scan_name else self.scan_root
        self.registration_root = self.work_dir / "02_registration"

    def variables(self):
        return {
            "ROOT_DIR": self.root_dir,
            "RUN_ROOT": self.run_root,
            "ACTIVE_ROOT": self.active_root,
            "WORK_DIR": self.work_dir,
            "RESULT_DIR": self.result_dir,
            "SCAN_ROOT": self.scan_root,
            "SCAN_NAME": self.scan_name,
            "SCAN_DIR": self.scan_dir,
            "REGISTRATION_ROOT": self.registration_root,
            "JOB_DIR": self.registration_root / "manual",
        }


def publish_cloud(source, destination):
    """成对发布 PCD 和旁文件；同盘硬链接，跨盘复制，不覆盖已有文件。"""
    source, destination = Path(source).resolve(), Path(destination).absolute()
    pairs = [(source, destination), (Path(str(source) + ".npz"), Path(str(destination) + ".npz"))]
    for src, dst in pairs:
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists() or dst.is_symlink():
            raise FileExistsError(dst)
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for src, dst in pairs:
            try:
                os.link(src, dst)
                created.append(dst)
            except OSError as error:
                if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                    raise
                with dst.open("xb") as output:
                    created.append(dst)
                    with src.open("rb") as input_file:
                        shutil.copyfileobj(input_file, output)
    except Exception:
        for path in created:
            path.unlink()
        raise
    return str(destination)
