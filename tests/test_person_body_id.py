"""Hardware-independent checks for the whole-person ReID application."""

import ast
from pathlib import Path

from hailo_apps.config.config_manager import get_default_model_name, get_model_names


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "hailo_apps/my_projects/auto_face_id/person_body_id.py"
API_SOURCE = ROOT / "hailo_apps/my_projects/auto_face_id/person_body_api.py"


def _body_app_class() -> ast.ClassDef:
    tree = ast.parse(SOURCE.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PersonBodyIdApp"
    )


def _method(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _body_app_class().body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _module_function(name: str) -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _called_attributes(method: ast.FunctionDef) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_hailo10h_body_reid_model_is_registered():
    assert get_default_model_name("person_body_id", "hailo10h", "pipeline") == (
        "repvgg_a0_person_reid_512"
    )
    assert get_model_names("person_body_id", "hailo10h", app_type="pipeline") == [
        "repvgg_a0_person_reid_512"
    ]


def test_body_app_does_not_resolve_face_resources():
    source = ast.unparse(_method("_resolve_identity_pipeline_resources"))
    assert "resolve_hef_paths" not in source
    assert "face_hef_path" not in source
    assert "body_reid_hef_path" in source


def test_entry_never_searches_historical_identities_and_exit_searches_inside_only():
    calls = _called_attributes(_method("_pipeline_callback_impl"))
    assert "_recognize_embedding" not in calls
    assert "_recognize_entered_embedding" in calls
    assert "_handle_unknown_person" in calls


def test_pipeline_prunes_to_people_before_every_frame_cropper():
    source = ast.unparse(_method("get_pipeline_string"))
    assert "body_person_filter_callback" in source
    assert "REID_CROPPER_POSTPROCESS_FUNCTION" in source
    assert source.rindex("body_person_filter_callback") < source.rindex("cropper")


def test_body_database_is_separate_from_face_embeddings():
    source = ast.unparse(_method("_build_body_parser"))
    assert "persons_body.sqlite3" in source
    assert "body_samples" in source
    assert "enable_watchdog=True" in source


def test_live_errors_and_eos_schedule_recovery_instead_of_shutdown():
    source = ast.unparse(_method("bus_call"))
    assert source.count("_schedule_pipeline_recovery") == 2
    assert "shutdown" not in source


def test_callback_exceptions_are_isolated_to_one_frame():
    source = ast.unparse(_method("pipeline_callback"))
    assert "except Exception" in source
    assert "Gst.FlowReturn.OK" in source


def test_stalled_hailo_worker_uses_process_restart_not_in_process_rebuild():
    source = ast.unparse(_method("_rebuild_pipeline"))
    assert "os._exit(RESTART_EXIT_CODE)" in source
    assert "super()._rebuild_pipeline" not in source


def test_supervisor_restarts_the_same_body_command():
    source = ast.unparse(_module_function("_run_supervisor"))
    assert "subprocess.Popen" in source
    assert "sys.argv[1:]" in source
    assert "BODY_WORKER_ENV" in source


def test_body_api_reexports_the_complete_face_api_contract_with_body_defaults():
    source = API_SOURCE.read_text()
    database_default = source.index('"PERSON_ID_DB_NAME", "persons_body.sqlite3"')
    sample_default = source.index('"PERSON_ID_SAMPLES_DIR", "body_samples"')
    shared_api_import = source.index("from hailo_apps.my_projects.auto_face_id.person_face_api import")

    assert database_default < shared_api_import
    assert sample_default < shared_api_import
    assert "    app," in source
    assert '"hailo_apps.my_projects.auto_face_id.person_body_api:app"' in source
