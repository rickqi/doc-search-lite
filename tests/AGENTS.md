# 测试 (lite)

## OVERVIEW

测试文件从企业版同步而来（80 个 .py 文件），但 **仅 `test_desensitization.py` 可独立运行**（31 tests）。
多数测试依赖企业版专有模块或需真实 Tantivy 索引。不要信任列出的测试数量。

## 实际可运行的测试

| 文件 | 测试数 | 状态 |
|------|--------|------|
| `test_desensitization.py` | 31 | ✅ 通过 |
| `test_utils_hash.py` | 21 | ✅ 通过 |
| `test_search_query_parser.py` | ~10 | ✅ 纯逻辑, 通过 |
| `test_chunker.py` | ~20 | ✅ 纯逻辑, 通过 |
| 其余 76 个文件 | — | ⚠️ 依赖缺失/真实索引/企业模块 |

## Fixtures (conftest.py)

| Fixture | 类型 | 说明 |
|---------|------|------|
| `mock_config` | function | `MagicMock(spec=Config)` — 含默认 test-key/model/temperature |
| `_env_snapshot` | autouse | 快照 `os.environ`，测试结束后自动恢复 |

## 关键约束

- `os.environ["KEY"] = ""` 非 `pop()` — `load_dotenv(override=False)` 重填充
- `GLM_API_KEY` + `GLM_BASE_URL` 为必需环境变量（`Config.from_env()` 检查）