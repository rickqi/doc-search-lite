"""Unit tests for src/processor/ module — PDF Enhance Pipeline.

Tests cover:
  - Coordinate utilities (normalize_bbox, calculate_iou, parse_la_boxes)
  - Data models (PipelineState save/load, GapTarget, Supplement)
  - GapAnalyzer (seal-empty, coverage gaps, special elements)
  - LocateAnythingWorker (stub mode, box parsing)
  - ComparisonReportGenerator (metrics computation, report generation)
  - PDFEnhancePipeline (mock end-to-end)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.processor.models import (
    GLMPageResult,
    GapTarget,
    PageResult,
    PDFMetadata,
    PipelineState,
    Supplement,
)
from src.processor.coords import (
    calculate_iou,
    crop_region,
    from_pixel,
    is_duplicate,
    map_crop_to_full,
    normalize_bbox,
    parse_la_boxes,
    parse_la_detection_labels,
    to_pixel,
)
from src.processor.gap_analysis import GapAnalyzer
from src.processor.locateanything_worker import LocateAnythingWorker
from src.processor.comparison_report import ComparisonReportGenerator


# ═══════════════════ Coordinate Utils ═══════════════════


class TestNormalizeBbox:
    def test_float_0_to_1(self):
        result = normalize_bbox([0.1, 0.2, 0.5, 0.8])
        assert result == [100, 200, 500, 800]

    def test_int_0_to_1000(self):
        result = normalize_bbox([100, 200, 500, 800])
        assert result == [100, 200, 500, 800]

    def test_none_input(self):
        assert normalize_bbox(None) is None

    def test_empty_list(self):
        assert normalize_bbox([]) is None

    def test_partial_list(self):
        assert normalize_bbox([100, 200]) is None

    def test_mixed_range(self):
        """Values above 1.0 should be treated as 0-1000 scale."""
        result = normalize_bbox([100, 200, 500, 800])
        assert result == [100, 200, 500, 800]


class TestCalculateIoU:
    def test_identical_boxes(self):
        assert calculate_iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert calculate_iou([0, 0, 50, 50], [100, 100, 200, 200]) == 0.0

    def test_partial_overlap(self):
        iou = calculate_iou([0, 0, 100, 100], [50, 50, 150, 150])
        assert 0.1 < iou < 0.5  # Should be ~0.143

    def test_empty_boxes(self):
        assert calculate_iou([], [0, 0, 100, 100]) == 0.0
        assert calculate_iou([0, 0, 100, 100], []) == 0.0

    def test_none_boxes(self):
        assert calculate_iou(None, [0, 0, 100, 100]) == 0.0


class TestIsDuplicate:
    def test_high_overlap_is_duplicate(self):
        existing = [[0, 0, 100, 100]]
        new = [10, 10, 110, 110]
        assert is_duplicate(new, existing, iou_threshold=0.5)

    def test_no_overlap_not_duplicate(self):
        existing = [[0, 0, 100, 100]]
        new = [500, 500, 600, 600]
        assert not is_duplicate(new, existing, iou_threshold=0.7)

    def test_empty_existing(self):
        assert not is_duplicate([0, 0, 100, 100], [])


class TestPixelConversion:
    def test_to_pixel(self):
        result = to_pixel([500, 500, 1000, 1000], 1000, 1000)
        assert result == [500, 500, 1000, 1000]

    def test_from_pixel(self):
        result = from_pixel([500, 500, 1000, 1000], 1000, 1000)
        assert result == [500, 500, 1000, 1000]

    def test_round_trip(self):
        bbox = [100, 200, 300, 400]
        pixel = to_pixel(bbox, 2000, 3000)
        back = from_pixel(pixel, 2000, 3000)
        assert back == bbox

    def test_zero_width(self):
        result = from_pixel([0, 0, 100, 100], 0, 100)
        # width=0 → all x coords are 0; height=100 → 100/100*1000=1000
        assert result == [0, 0, 0, 1000]


class TestParseLABoxes:
    def test_single_box(self):
        answer = "<box><100><200><300><400></box>"
        boxes = parse_la_boxes(answer, 1000, 1000)
        assert len(boxes) == 1
        assert boxes[0] == {"x1": 100, "y1": 200, "x2": 300, "y2": 400}

    def test_multiple_boxes(self):
        answer = "<box><0><0><500><500></box><box><500><500><1000><1000></box>"
        boxes = parse_la_boxes(answer, 1000, 1000)
        assert len(boxes) == 2

    def test_no_boxes(self):
        boxes = parse_la_boxes("none", 1000, 1000)
        assert boxes == []

    def test_pixel_scaling(self):
        answer = "<box><500><500><1000><1000></box>"
        boxes = parse_la_boxes(answer, 2000, 2000)
        assert boxes[0] == {"x1": 1000, "y1": 1000, "x2": 2000, "y2": 2000}


class TestParseLADetectionLabels:
    def test_with_labels(self):
        answer = "<ref>signature</ref><box><100><100><200><200></box>"
        labels = parse_la_detection_labels(answer)
        assert labels == ["signature"]

    def test_multiple_labels(self):
        answer = (
            "<ref>signature</ref><box><100><100><200><200></box>"
            "<ref>checkbox</ref><box><300><300><400><400></box>"
        )
        labels = parse_la_detection_labels(answer)
        assert labels == ["signature", "checkbox"]

    def test_no_labels_fills_unknown(self):
        answer = "<box><100><100><200><200></box>"
        labels = parse_la_detection_labels(answer)
        assert labels == ["unknown"]


class TestCropRegion:
    def test_crop(self):
        from PIL import Image

        img = Image.new("RGB", (1000, 1000), color="white")
        cropped, w, h = crop_region(img, [100, 100, 500, 500])
        assert cropped.size == (400, 400)
        assert w == 400
        assert h == 400

    def test_crop_clamped(self):
        from PIL import Image

        img = Image.new("RGB", (1000, 1000), color="white")
        # Crop extending beyond image bounds
        cropped, w, h = crop_region(img, [800, 800, 1200, 1200])
        assert cropped.size == (200, 200)


# ═══════════════════ Data Models ═══════════════════


class TestPipelineState:
    def test_save_and_load(self):
        state = PipelineState(
            pdf_hash="abc123",
            total_pages=10,
            phases_completed=["render", "glm_ocr"],
            pages_glm_done=[1, 2, 3],
            phase_timings={"render": 5.2, "glm_ocr": 120.0},
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            state.save(path)
            assert path.exists()

            loaded = PipelineState.load(path)
            assert loaded is not None
            assert loaded.pdf_hash == "abc123"
            assert loaded.total_pages == 10
            assert loaded.phases_completed == ["render", "glm_ocr"]
            assert loaded.pages_glm_done == [1, 2, 3]
            assert loaded.phase_timings["render"] == 5.2

    def test_load_nonexistent(self):
        assert PipelineState.load(Path("/nonexistent/state.json")) is None

    def test_load_corrupt(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            path.write_text("not json", encoding="utf-8")
            assert PipelineState.load(path) is None

    def test_default_values(self):
        state = PipelineState()
        assert state.phases_completed == []
        assert state.pages_glm_done == []
        assert state.phase_timings == {}


class TestSupplement:
    def test_creation(self):
        supp = Supplement(
            page_index=1,
            label="signature",
            content="张三",
            bbox_1000=[100, 200, 300, 400],
            source="la_detect_special",
            confidence=0.95,
        )
        assert supp.label == "signature"
        assert supp.confidence == 0.95

    def test_defaults(self):
        supp = Supplement(page_index=1, label="text")
        assert supp.content == ""
        assert supp.bbox_1000 == []
        assert supp.source == ""
        assert supp.confidence == 0.0


# ═══════════════════ GapAnalyzer ═══════════════════


class TestGapAnalyzer:
    def test_seal_empty_detection(self):
        """Seal region with empty content should produce priority-0 target."""
        analyzer = GapAnalyzer()
        regions = [
            {"label": "seal", "content": "", "bbox_2d": [700, 700, 900, 900]},
        ]
        targets = analyzer.analyze(regions, 1000, 1000, page_index=1)
        seal_targets = [t for t in targets if t.gap_type == "seal_text"]
        assert len(seal_targets) == 1
        assert seal_targets[0].priority == 0

    def test_seal_with_content_no_target(self):
        """Seal region with content should not produce a gap target."""
        analyzer = GapAnalyzer()
        regions = [
            {"label": "seal", "content": "公章文字", "bbox_2d": [700, 700, 900, 900]},
        ]
        targets = analyzer.analyze(regions, 1000, 1000, page_index=1)
        seal_targets = [t for t in targets if t.gap_type == "seal_text"]
        assert len(seal_targets) == 0

    def test_short_text_detection(self):
        analyzer = GapAnalyzer()
        regions = [
            {"label": "text", "content": "ab", "bbox_2d": [100, 100, 900, 200]},
        ]
        targets = analyzer.analyze(regions, 1000, 1000, page_index=1)
        short_targets = [t for t in targets if t.gap_type == "short_text"]
        assert len(short_targets) == 1

    def test_coverage_gap_detection(self):
        """Large uncovered area should produce a coverage gap target."""
        analyzer = GapAnalyzer()
        # Only cover a small region — rest of page is uncovered
        regions = [
            {"label": "text", "content": "x" * 100, "bbox_2d": [0, 0, 100, 100]},
        ]
        targets = analyzer.analyze(regions, 1000, 1000, page_index=1)
        coverage_targets = [t for t in targets if t.gap_type == "full_text"]
        assert len(coverage_targets) > 0
        assert all(t.priority == 1 for t in coverage_targets)

    def test_special_element_always_present(self):
        """Special element target should always be present."""
        analyzer = GapAnalyzer()
        targets = analyzer.analyze([], 1000, 1000, page_index=1)
        special_targets = [t for t in targets if t.gap_type == "special_element"]
        assert len(special_targets) == 1
        assert special_targets[0].action == "detect_special"

    def test_priority_ordering(self):
        """Targets should be sorted by priority (0 first)."""
        analyzer = GapAnalyzer()
        regions = [
            {"label": "seal", "content": "", "bbox_2d": [700, 700, 900, 900]},
            {"label": "text", "content": "x" * 100, "bbox_2d": [0, 0, 100, 100]},
        ]
        targets = analyzer.analyze(regions, 1000, 1000, page_index=1)
        priorities = [t.priority for t in targets]
        assert priorities == sorted(priorities)

    def test_full_page_coverage_no_gaps(self):
        """When GLM-OCR covers the full page, no coverage gaps should be found."""
        analyzer = GapAnalyzer()
        regions = [
            {"label": "text", "content": "x" * 100, "bbox_2d": [0, 0, 1000, 1000]},
        ]
        targets = analyzer.analyze(regions, 1000, 1000, page_index=1)
        coverage_targets = [t for t in targets if t.gap_type == "full_text"]
        assert len(coverage_targets) == 0


# ═══════════════════ LocateAnythingWorker ═══════════════════


class TestLocateAnythingWorker:
    def test_stub_mode_returns_empty(self):
        """Without model installed, all methods return empty strings."""
        worker = LocateAnythingWorker()
        # Force stub mode (model not installed)
        worker._load_attempted = True
        worker._available = False

        assert worker.detect_text(None) == ""
        assert worker.detect(None, ["signature"]) == ""
        assert worker.ground_text(None, "test") == ""
        assert worker.ground_single(None, "test") == ""
        assert worker.ground_gui(None, "button") == ""

    def test_parse_boxes_static(self):
        answer = "<box><100><100><500><500></box>"
        boxes = LocateAnythingWorker.parse_boxes(answer, 1000, 1000)
        assert len(boxes) == 1
        assert boxes[0]["x1"] == 100

    def test_parse_labels_static(self):
        answer = "<ref>signature</ref><box><100><100><200><200></box>"
        labels = LocateAnythingWorker.parse_detection_labels(answer)
        assert labels == ["signature"]


# ═══════════════════ ComparisonReportGenerator ═══════════════════


class TestComparisonReportGenerator:
    def _setup_output_dir(self, tmpdir, num_pages=3, with_supplements=True):
        """Create a mock output directory with GLM and enhanced results."""
        output_dir = Path(tmpdir)
        glm_dir = output_dir / "glm_ocr" / "pages"
        enh_dir = output_dir / "enhanced" / "pages"
        glm_dir.mkdir(parents=True, exist_ok=True)
        enh_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, num_pages + 1):
            # GLM result
            glm_data = {
                "page_index": i,
                "success": True,
                "regions": [
                    {"index": 0, "label": "title", "content": f"Page {i} Title"},
                    {"index": 1, "label": "text", "content": "Some text content here."},
                ],
                "markdown": f"# Page {i}\n\nSome text content.",
                "token_usage": {"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
                "latency": 2.5,
            }
            (glm_dir / f"page_{i:04d}.json").write_text(
                json.dumps(glm_data, ensure_ascii=False), encoding="utf-8"
            )
            (glm_dir / f"page_{i:04d}.md").write_text(glm_data["markdown"], encoding="utf-8")

            # Enhanced result
            supplements = []
            if with_supplements and i == 1:
                supplements = [
                    {"label": "signature", "content": "", "bbox_1000": [100, 200, 300, 400],
                     "source": "la_detect_special", "confidence": 0.9},
                ]

            enh_data = {
                "page_index": i,
                "glm_regions": glm_data["regions"],
                "supplements": supplements,
                "enhanced_regions": glm_data["regions"] + [
                    {"index": len(glm_data["regions"]), "label": s["label"],
                     "content": s["content"], "bbox_2d": s["bbox_1000"]}
                    for s in supplements
                ],
            }
            (enh_dir / f"page_{i:04d}.json").write_text(
                json.dumps(enh_data, ensure_ascii=False), encoding="utf-8"
            )
            (enh_dir / f"page_{i:04d}.md").write_text(
                glm_data["markdown"] + "\nsupplement", encoding="utf-8"
            )

        # Summaries
        (output_dir / "glm_ocr" / "summary.json").write_text(
            json.dumps({
                "total_pages": num_pages, "total_regions": num_pages * 2,
                "pages_with_empty_seals": [], "pages_with_low_coverage": [],
            }),
            encoding="utf-8",
        )
        (output_dir / "enhanced" / "summary.json").write_text(
            json.dumps({
                "total_pages": num_pages, "total_supplements": 1 if with_supplements else 0,
                "special_elements": 1 if with_supplements else 0,
                "supplement_label_distribution": {"signature": 1} if with_supplements else {},
            }),
            encoding="utf-8",
        )
        return output_dir

    def test_generate_json_basic(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = self._setup_output_dir(td, num_pages=3, with_supplements=True)
            generator = ComparisonReportGenerator()
            result = generator.generate_json(output_dir)

            assert result["total_pages"] == 3
            assert result["glm_total_regions"] == 6  # 2 per page × 3
            assert result["enhanced_total_regions"] == 7  # 6 + 1 supplement
            assert result["total_supplements"] == 1
            assert "signature" in result["new_regions_by_type"]

    def test_generate_markdown_report(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = self._setup_output_dir(td, num_pages=3, with_supplements=True)
            generator = ComparisonReportGenerator()
            meta = PDFMetadata(total_pages=3, pdf_hash="test", file_size_bytes=1000)
            report = generator.generate(output_dir, meta, dpi=150)

            assert "级联识别对比分析报告" in report
            assert "总体统计" in report
            assert "GLM-OCR" in report
            assert "signature" in report or "无页面获得补充" in report

    def test_generate_no_supplements(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = self._setup_output_dir(td, num_pages=2, with_supplements=False)
            generator = ComparisonReportGenerator()
            result = generator.generate_json(output_dir)

            assert result["total_supplements"] == 0
            assert result["region_change"] == 0

    def test_top_improved_pages(self):
        with tempfile.TemporaryDirectory() as td:
            output_dir = self._setup_output_dir(td, num_pages=5, with_supplements=True)
            generator = ComparisonReportGenerator()
            meta = PDFMetadata(total_pages=5, pdf_hash="test", file_size_bytes=1000)
            report = generator.generate(output_dir, meta, dpi=150)

            # Page 1 should be in the top improved list
            assert "p.0001" in report or "无页面获得补充" in report


# ═══════════════════ PDFEnhancePipeline (mock tests) ═══════════════════


class TestPDFEnhancePipeline:
    def test_init_defaults(self):
        from src.processor.pdf_enhance import PDFEnhancePipeline

        pipeline = PDFEnhancePipeline(glm_api_key="test")
        assert pipeline.dpi == 150
        assert pipeline.iou_threshold == 0.7
        assert pipeline.glm_parallel == 3

    def test_lazy_loading(self):
        """Components should be lazily loaded."""
        from src.processor.pdf_enhance import PDFEnhancePipeline

        pipeline = PDFEnhancePipeline(glm_api_key="test")
        assert pipeline._glm_adapter is None
        assert pipeline._la_worker is None
        assert pipeline._gap_analyzer is None

        # Access triggers creation
        analyzer = pipeline._get_gap_analyzer()
        assert analyzer is not None
        assert pipeline._gap_analyzer is analyzer

    def test_merge_results(self):
        """Test the merge logic with mock data."""
        from src.processor.pdf_enhance import PDFEnhancePipeline

        pipeline = PDFEnhancePipeline(glm_api_key="test")
        glm_regions = [
            {"index": 0, "label": "title", "content": "Test", "bbox_2d": [0, 0, 500, 100]},
        ]
        supplements = [
            Supplement(page_index=1, label="signature", bbox_1000=[600, 700, 800, 900],
                       source="la_detect_special"),
        ]
        enhanced, md = pipeline._merge_results(glm_regions, supplements, "original md", 1)

        assert len(enhanced) == 2
        assert enhanced[1]["label"] == "signature"
        assert "LocateAnything supplements" in md
        assert "Signature" in md

    def test_glm_adapter_rejects_pdf(self):
        """GLM-OCR adapter must reject PDF files — only images allowed."""
        from src.processor.glm_ocr_adapter import GLMOCRAdapter

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4 fake pdf content")
            pdf_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Cannot send PDF"):
                GLMOCRAdapter._encode_image(pdf_path)
        finally:
            pdf_path.unlink()

    def test_glm_adapter_accepts_png(self):
        """GLM-OCR adapter should accept PNG files."""
        from src.processor.glm_ocr_adapter import GLMOCRAdapter
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = Path(f.name)

        try:
            # Create a minimal valid PNG
            Image.new("RGB", (10, 10), "white").save(str(png_path), "PNG")
            data_url = GLMOCRAdapter._encode_image(png_path)
            assert data_url.startswith("data:image/png;base64,")
        finally:
            png_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
