from pathlib import Path


def test_dashboard_draws_risk_grid_with_its_own_metadata():
    source = (
        Path(__file__).parents[1] / 'static' / 'dashboard.js'
    ).read_text(encoding='utf-8')

    assert 'function cellToWorld(meta, cellX, cellY)' in source
    assert 'function drawGridImage(img, overlayMeta, baseMeta, vp)' in source
    assert 'drawGridImage(riskImg, latest.risk.metadata, meta, vp)' in source
    assert 'ctx.drawImage(riskImg, vp.x, vp.y, vp.w, vp.h)' not in source
