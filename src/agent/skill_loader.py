"""Load external SKILL.md files and inject into agent system prompts.

Skill discovery supports two layouts:

1. **Directory layout** (Agent Skills standard): ``<dir>/<skill_name>/SKILL.md``
2. **Flat layout**: ``<dir>/<skill_name>.md`` (standalone markdown files)

Search paths (in priority order):
    - Project-local ``skills/`` directory
    - ``~/.agents/skills/``
    - ``~/.opencode/skills/``
    - ``SKILLS_PATH`` environment variable (colon/semicolon separated)
"""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_search_paths() -> list[Path]:
    """Build cross-platform skill search paths.

    Priority order:
        1. ``SKILLS_PATH`` environment variable (explicit override)
        2. Project-local ``skills/`` directory (relative to this file's project root)
        3. ``~/.agents/skills/``
        4. ``~/.opencode/skills/``
    """
    paths: list[Path] = []

    env_val = os.environ.get("SKILLS_PATH", "")
    if env_val:
        sep = ";" if os.pathsep == ";" else ":"
        for p in env_val.split(sep):
            p = p.strip()
            if p:
                paths.append(Path(p))

    project_root = Path(__file__).resolve().parent.parent.parent
    paths.append(project_root / "skills")

    paths.append(Path.home() / ".agents" / "skills")
    paths.append(Path.home() / ".opencode" / "skills")

    return paths


def discover_skill_dirs() -> list[Path]:
    """Find all directories that may contain skills."""
    return [p for p in _get_search_paths() if p.is_dir()]


def list_available_skills() -> dict[str, str]:
    """List all available skills with their descriptions.

    Supports both directory layout (``<name>/SKILL.md``) and flat layout
    (``<name>.md``).

    Returns:
        dict mapping skill_name -> short description
    """
    skills: dict[str, str] = {}
    for skill_dir in discover_skill_dirs():
        for child in sorted(skill_dir.iterdir()):
            # Directory layout: <name>/SKILL.md
            if child.is_dir():
                skill_md = child / "SKILL.md"
                if skill_md.exists():
                    desc = _extract_description(skill_md)
                    skills[child.name] = desc
            # Flat layout: <name>.md
            elif child.is_file() and child.suffix == ".md":
                if child.name != "_index.md":
                    desc = _extract_description(child)
                    skills[child.stem] = desc
    return skills


def load_skill_content(skill_name: str) -> Optional[str]:
    """Load a SKILL.md file by skill name or direct file path.

    Lookup order:
        1. ``<search_dir>/<skill_name>/SKILL.md`` (directory layout)
        2. ``<search_dir>/<skill_name>.md`` (flat layout)
        3. Direct file path if ``skill_name`` looks like a path

    Args:
        skill_name: Name of the skill (directory name) or direct file path

    Returns:
        Skill content string, or None if not found
    """
    # Search standard skill directories
    for skill_dir in discover_skill_dirs():
        # Directory layout
        skill_md = skill_dir / skill_name / "SKILL.md"
        if skill_md.exists():
            try:
                content = skill_md.read_text(encoding="utf-8")
                logger.info("Loaded skill '%s' from %s", skill_name, skill_md)
                return content
            except Exception as e:
                logger.warning("Failed to read skill %s: %s", skill_md, e)

        # Flat layout
        flat_md = skill_dir / f"{skill_name}.md"
        if flat_md.exists():
            try:
                content = flat_md.read_text(encoding="utf-8")
                logger.info("Loaded skill '%s' from %s", skill_name, flat_md)
                return content
            except Exception as e:
                logger.warning("Failed to read skill %s: %s", flat_md, e)

    # Direct file path fallback
    direct_path = Path(skill_name)
    if direct_path.exists() and direct_path.suffix == ".md":
        try:
            content = direct_path.read_text(encoding="utf-8")
            logger.info("Loaded skill from direct path: %s", direct_path)
            return content
        except Exception as e:
            logger.warning("Failed to read skill file %s: %s", direct_path, e)

    logger.warning("Skill '%s' not found in any search path", skill_name)
    return None


def _extract_description(skill_md: Path) -> str:
    """Extract short description from SKILL.md frontmatter or first paragraph."""
    try:
        content = skill_md.read_text(encoding="utf-8")
        lines = content.split("\n")
        # Look for description in YAML frontmatter
        in_frontmatter = False
        for line in lines:
            stripped = line.strip()
            if stripped == "---":
                if in_frontmatter:
                    break
                in_frontmatter = True
                continue
            if in_frontmatter and stripped.startswith("description:"):
                desc = stripped[len("description:"):].strip().strip('"').strip("'")
                if desc:
                    return desc[:100]
        # Fallback: first non-empty, non-heading line
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("---"):
                return stripped[:100]
        return ""
    except Exception:
        return ""
