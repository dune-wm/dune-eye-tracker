import json

from dune_eye_tracker import clicks


def test_a_click_while_zoomed_is_learnt_where_it_was_seen():
    line = {"event": "pointer", "change": "click", "button": "left", "x": 820, "y": 390, "output": "M",
            "tile": 3, "app_id": "foot", "tile_x": 814, "tile_y": 384, "tile_w": 938, "tile_h": 1054,
            "zoom_x": 860, "zoom_y": 370, "zoom_scale": 3}
    x, y, ctx = clicks.seen(line)
    assert (x, y) == (860, 370)  # the eyes were on the magnified point
    assert ctx["real_x"] == 820 and ctx["real_y"] == 390 and ctx["zoom_scale"] == 3
    assert ctx["app_id"] == "foot" and ctx["tile_x"] == 814  # the window's, real
    plain = {k: v for k, v in line.items() if not k.startswith("zoom")}
    assert clicks.seen(plain)[:2] == (820, 390)


def test_zoom_lines():
    zin = {"event": "zoom", "change": "in", "output": "M", "x": 800, "y": 400, "scale": 3, "tile": 1,
           "area_x": 6, "area_y": 6, "area_w": 938, "area_h": 1054}
    assert clicks.zoom_of(zin) == {"x": 800, "y": 400, "scale": 3, "area_x": 6, "area_y": 6, "area_w": 938, "area_h": 1054}
    assert clicks.zoom_of({"event": "zoom", "change": "out", "output": "M"}) is None


def test_the_voice_counts_as_much_as_the_mouse_and_a_key_less():
    assert clicks.weight("mouse") == clicks.weight("command") == clicks.weight("name") == 1.0
    assert 0 < clicks.weight("key") < 1.0
    assert clicks.weight(None) == 1.0


def test_the_focused_tile_is_followed_from_dunes_events():
    f = clicks.Focus()
    assert f.at("M", 0.0) is None
    f.apply({"event": "strip", "change": "layout", "output": "M", "view": 1282, "width": 2560, "height": 1080,
             "tiles": [{"tile": 3, "x": 8, "y": 8, "w": 1268, "h": 1064}, {"tile": 5, "x": 1290, "y": 8, "w": 1268, "h": 1064}]}, 0.0)
    f.apply({"event": "window", "change": "focus", "tile": 3, "app_id": "foot", "title": "~"}, 1.0)
    f.apply({"event": "window", "change": "focus", "tile": 5, "app_id": "foot", "title": "~"}, 2.0)
    assert f.at("M", 2.5) == {"tile": 5, "prev": 3, "age": 0.5, "x": 8, "y": 8, "w": 1268, "h": 1064}
    assert f.at("other", 2.5) == {"tile": 5, "prev": 3, "age": 0.5}  # not on that monitor's strip
    f.apply({"event": "window", "change": "focus", "tile": 5}, 3.0)  # the same again: not a change
    assert f.at("M", 3.0)["age"] == 1.0
    f.apply({"event": "window", "change": "close", "tile": 5}, 4.0)
    assert f.at("M", 4.0) is None


def test_what_made_a_click_is_heard():
    line = {"event": "pointer", "change": "click", "button": "left", "by": "key", "x": 1, "y": 2, "output": "M"}
    assert clicks.how(line) == {"by": "key", "button": "left"}
    assert clicks.how({"event": "pointer", "change": "click", "button": "right", "x": 1, "y": 2}) == {"button": "right"}


def test_a_log_line_has_the_time_first_and_survives_a_bad_path(tmp_path):
    clicks.log(tmp_path / clicks.LOG, {"seen": [1, 2], "by": "mouse"})
    line = json.loads((tmp_path / clicks.LOG).read_text())
    assert list(line)[0] == "time" and len(line["time"]) == len("2026-11-03T10:41:13.123") and line["by"] == "mouse"
    clicks.log(tmp_path / "missing" / clicks.LOG, {"seen": [1, 2]})  # no folder: nothing, no error
