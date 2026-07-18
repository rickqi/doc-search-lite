"""独立 raw 目录存储，与源目录完全隔离。

映射规则:
    source:      /path/to/source/my-data/subdir/file.xlsx
    source_root: /path/to/source/my-data
    raw_root:    /path/to/raw
    -> /path/to/raw/my-data/subdir/file.xlsx.md
    -> /path/to/raw/my-data/subdir/file.xlsx.md.json
    -> /path/to/raw/my-data/subdir/file.xlsx_images/  (images)
"""

import json
import shutil
from pathlib import Path


class RawStore:
    """独立 raw 目录存储，与源目录完全隔离。

    映射规则:
        source: /path/to/source/my-data/subdir/file.xlsx
        source_root: /path/to/source/my-data
        raw_root: /path/to/raw
        -> /path/to/raw/my-data/subdir/file.xlsx.md
    """

    METADATA_SUFFIX = ".md.json"
    INDEX_FILENAME = "_index.md"

    def __init__(self, source_root: Path, raw_root: Path):
        """
        Args:
            source_root: 源目录根路径 (e.g. /path/to/source/my-data)
            raw_root: 输出 raw 根路径 (e.g. /path/to/raw)
        """
        self.source_root = Path(source_root).resolve()
        self.raw_root = Path(raw_root).resolve()

    # ── Path Mapping ──────────────────────────────

    def get_output_root(self) -> Path:
        """返回 {raw_root}/{source_root_name}。

        Returns:
            输出根路径，例如 /path/to/raw/my-data
        """
        return self.raw_root / self.source_root.name

    def map_output_path(self, source_file: Path) -> Path:
        """将源文件映射为输出 .md 路径。

        source_file 相对于 source_root 的路径，保留原扩展名追加 .md。

        Args:
            source_file: 源文件路径

        Returns:
            输出 Markdown 文件路径 (e.g. file.xlsx.md)
        """
        source_file = Path(source_file).resolve()
        rel = source_file.relative_to(self.source_root)
        output_root = self.get_output_root()
        # 保留原扩展名: file.xlsx -> file.xlsx.md
        return output_root / Path(str(rel) + ".md")

    def map_metadata_path(self, source_file: Path) -> Path:
        """将源文件映射为输出 .md.json 元数据路径。

        Args:
            source_file: 源文件路径

        Returns:
            输出元数据 JSON 文件路径
        """
        md_path = self.map_output_path(source_file)
        return Path(str(md_path) + ".json")

    def map_index_path(self, source_dir: Path) -> Path:
        """将源目录映射为 _index.md 路径。

        Args:
            source_dir: 源目录路径

        Returns:
            输出 _index.md 文件路径
        """
        source_dir = Path(source_dir).resolve()
        rel = source_dir.relative_to(self.source_root)
        output_root = self.get_output_root()
        return output_root / rel / self.INDEX_FILENAME

    def map_images_dir(self, source_file: Path) -> Path:
        """将源文件映射为图片输出目录。

        每个文件拥有独立的 {stem}_images/ 目录。

        Args:
            source_file: 源文件路径

        Returns:
            图片输出目录路径
        """
        md_path = self.map_output_path(source_file)
        return md_path.parent / (md_path.stem + "_images")

    # ── Save Operations ───────────────────────────

    def save(self, source_file: Path, markdown: str, metadata: dict) -> Path:
        """保存转换后的 Markdown 和元数据。

        自动创建所需目录结构。

        Args:
            source_file: 源文件路径
            markdown: Markdown 内容
            metadata: 元数据字典

        Returns:
            输出 Markdown 文件路径
        """
        md_path = self.map_output_path(source_file)
        meta_path = self.map_metadata_path(source_file)

        # 创建父目录
        md_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入 Markdown
        md_path.write_text(markdown, encoding="utf-8")

        # 写入元数据 JSON
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return md_path

    def save_with_images(
        self,
        source_file: Path,
        markdown: str,
        metadata: dict,
        images: list[Path],
    ) -> Path:
        """保存 Markdown + 元数据 + 复制图片到 {stem}_images/ 目录。

        Args:
            source_file: 源文件路径
            markdown: Markdown 内容
            metadata: 元数据字典
            images: 图片文件路径列表

        Returns:
            输出 Markdown 文件路径
        """
        md_path = self.save(source_file, markdown, metadata)

        # 复制图片
        if images:
            images_dir = self.map_images_dir(source_file)
            images_dir.mkdir(parents=True, exist_ok=True)

            for img_path in images:
                img_path = Path(img_path)
                if img_path.exists():
                    dest = images_dir / img_path.name
                    shutil.copy2(img_path, dest)

        return md_path

    # ── Read Operations ───────────────────────────

    def load_markdown(self, source_file: Path) -> str | None:
        """加载转换后的 Markdown 内容。

        Args:
            source_file: 源文件路径

        Returns:
            Markdown 内容字符串，不存在则返回 None
        """
        md_path = self.map_output_path(source_file)
        if not md_path.exists():
            return None
        return md_path.read_text(encoding="utf-8")

    def load_metadata(self, source_file: Path) -> dict | None:
        """加载元数据 JSON。

        Args:
            source_file: 源文件路径

        Returns:
            元数据字典，不存在则返回 None
        """
        meta_path = self.map_metadata_path(source_file)
        if not meta_path.exists():
            return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def load_by_output(self, output_path: Path) -> tuple[str, dict] | None:
        """通过输出 .md 路径直接加载 Markdown 和元数据。

        Args:
            output_path: 输出 .md 文件路径

        Returns:
            (markdown, metadata) 元组，不存在则返回 None
        """
        output_path = Path(output_path)
        if not output_path.exists():
            return None

        meta_path = Path(str(output_path) + ".json")
        if not meta_path.exists():
            return None

        try:
            markdown = output_path.read_text(encoding="utf-8")
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            return (markdown, metadata)
        except (json.JSONDecodeError, OSError):
            return None

    # ── Existence Checks ──────────────────────────

    def exists(self, source_file: Path) -> bool:
        """检查源文件是否有转换输出。

        Args:
            source_file: 源文件路径

        Returns:
            存在返回 True
        """
        return self.map_output_path(source_file).exists()

    def output_exists(self, output_path: Path) -> bool:
        """检查指定输出路径是否存在。

        Args:
            output_path: 输出文件路径

        Returns:
            存在返回 True
        """
        return Path(output_path).exists()

    # ── Delete Operations ─────────────────────────

    def delete(self, source_file: Path) -> bool:
        """删除源文件对应的转换输出 (.md + .md.json + 空图片目录)。

        Args:
            source_file: 源文件路径

        Returns:
            删除成功返回 True
        """
        md_path = self.map_output_path(source_file)
        meta_path = self.map_metadata_path(source_file)
        images_dir = self.map_images_dir(source_file)

        deleted = False

        if md_path.exists():
            md_path.unlink()
            deleted = True

        if meta_path.exists():
            meta_path.unlink()
            deleted = True

        # 删除图片目录（含内容）
        if images_dir.exists():
            shutil.rmtree(images_dir)
            deleted = True

        return deleted

    def delete_output(self, output_path: Path) -> bool:
        """通过输出路径直接删除。

        Args:
            output_path: 输出 .md 文件路径

        Returns:
            删除成功返回 True
        """
        output_path = Path(output_path)
        if not output_path.exists():
            return False

        meta_path = Path(str(output_path) + ".json")
        images_dir = output_path.parent / (output_path.stem + "_images")

        # 删除 .md
        output_path.unlink()

        # 删除 .md.json
        if meta_path.exists():
            meta_path.unlink()

        # 删除图片目录
        if images_dir.exists():
            shutil.rmtree(images_dir)

        return True

    # ── Listing ───────────────────────────────────

    def list_outputs(self) -> list[Path]:
        """列出输出目录树中所有 .md 文件。

        Returns:
            .md 文件路径列表
        """
        output_root = self.get_output_root()
        if not output_root.exists():
            return []
        return sorted(output_root.rglob("*.md"))

    def list_directories(self) -> list[Path]:
        """列出输出目录树中所有子目录。

        Returns:
            目录路径列表
        """
        output_root = self.get_output_root()
        if not output_root.exists():
            return []
        return sorted(d for d in output_root.rglob("*") if d.is_dir())

    # ── Utility ───────────────────────────────────

    def get_source_relative(self, source_file: Path) -> Path:
        """获取源文件相对于 source_root 的相对路径。

        Args:
            source_file: 源文件路径

        Returns:
            相对路径
        """
        source_file = Path(source_file).resolve()
        return source_file.relative_to(self.source_root)

    def resolve_source_file(self, relative_path: str) -> Path:
        """将相对路径解析为绝对源文件路径。

        Args:
            relative_path: 相对路径字符串

        Returns:
            绝对源文件路径
        """
        return self.source_root / relative_path

    def get_disk_usage(self) -> dict:
        """获取输出目录的总大小和文件计数。

        Returns:
            {"total_size": int, "file_count": int, "md_count": int}
        """
        output_root = self.get_output_root()
        total_size = 0
        file_count = 0
        md_count = 0

        if output_root.exists():
            for f in output_root.rglob("*"):
                if f.is_file():
                    try:
                        total_size += f.stat().st_size
                    except OSError:
                        pass
                    file_count += 1
                    if f.suffix == ".md":
                        md_count += 1

        return {
            "total_size": total_size,
            "file_count": file_count,
            "md_count": md_count,
        }
