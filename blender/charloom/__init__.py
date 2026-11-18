"""Charloom character assembly and editing in Blender."""

import os
from datetime import datetime
from pathlib import Path

import bpy
from bpy.props import EnumProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import api


def absolute(value):
    return Path(bpy.path.abspath(value)).expanduser()


class CharloomPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    checkout: StringProperty(
        name="Charloom 工程目录",
        subtype="DIR_PATH",
        default=os.getenv("CHARLOOM_ROOT", ""),
    )
    python: StringProperty(name="CV Python（可选）", subtype="FILE_PATH")
    bun: StringProperty(name="Bun（可选）", subtype="FILE_PATH")

    def draw(self, context):
        for key in ("checkout", "python", "bun"):
            self.layout.prop(self, key)


class CharloomSettings(bpy.types.PropertyGroup):
    source_mode: EnumProperty(
        name="输入",
        items=[("TRANSFER", "标定迁移目录", ""), ("SCAN", "原始扫描 GLB", "")],
    )
    source_file: StringProperty(name="扫描 GLB", subtype="FILE_PATH")
    prior: StringProperty(name="MakeHuman 模板", subtype="FILE_PATH")
    guidance: StringProperty(name="标注 JSON", subtype="FILE_PATH")
    mouth_trace: StringProperty(name="口缝标注", subtype="FILE_PATH")
    motion: StringProperty(name="行走动作", subtype="FILE_PATH")
    name: StringProperty(name="角色配置名称")
    segment_review: StringProperty(name="分区决定", subtype="FILE_PATH")
    segment_evidence: StringProperty(name="已审阅的三角形范围", subtype="FILE_PATH")
    transfer: StringProperty(name="标定好的迁移目录", subtype="DIR_PATH")
    output: StringProperty(name="输出父目录", subtype="DIR_PATH")
    config: StringProperty(name="角色配置（可选）", subtype="FILE_PATH")
    stage: EnumProperty(
        name="导入版本",
        items=[
            ("4-garments/character.glb", "穿衣", ""),
            ("3-skin/body.glb", "裸模", ""),
        ],
    )
    last_build: StringProperty(name="最近构建", subtype="DIR_PATH")
    rig: PointerProperty(
        name="骨架",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "ARMATURE",
    )
    body: PointerProperty(
        name="身体", type=bpy.types.Object, poll=lambda self, obj: obj.type == "MESH"
    )
    scan: PointerProperty(
        name="可见皮肤参考",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
    )
    group: StringProperty(name="拟合顶点组")


class CHARLOOM_OT_build(bpy.types.Operator):
    bl_idname = "charloom.build"
    bl_label = "从扫描重建原头绑定"
    bl_description = "使用标定文件和已拟合身体生成裸模、穿衣版及动画检查"
    _process = None
    _timer = None

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        settings = context.scene.charloom
        try:
            if not prefs.checkout:
                raise ValueError("在插件偏好设置中选择 Charloom 工程目录")
            root = absolute(prefs.checkout).resolve()
            if not settings.output:
                raise ValueError("选择输出父目录")
            options = {}
            if settings.source_mode == "SCAN":
                if not all((settings.source_file, settings.prior, settings.guidance)):
                    raise ValueError("选择原扫描、MakeHuman 模板和标注文件")
                transfer = absolute(settings.source_file).resolve()
                options = {
                    k: absolute(getattr(settings, k)).resolve()
                    for k in (
                        "prior",
                        "guidance",
                        "mouth_trace",
                        "motion",
                        "segment_review",
                        "segment_evidence",
                    )
                    if getattr(settings, k)
                }
                options["name"] = settings.name or transfer.stem
            else:
                transfer = absolute(settings.transfer).resolve()
                if not (transfer / "0-guide/resolved.json").is_file():
                    raise ValueError("迁移目录缺少 0-guide/resolved.json")
            parent = absolute(settings.output).resolve()
            parent.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            self._destination = parent / f"{transfer.stem}-{stamp}"
            self._scene = context.scene
            self._stage = settings.stage
            self._log = self._destination.with_suffix(".log")
            with self._log.open("x") as log:
                self._process = api.start_build(
                    transfer,
                    self._destination,
                    checkout=root,
                    log=log,
                    **options,
                    config=absolute(settings.config) if settings.config else None,
                    python=absolute(prefs.python) if prefs.python else None,
                    bun=absolute(prefs.bun) if prefs.bun else None,
                )
        except (OSError, ValueError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        if bpy.app.background:
            self._process.wait()
            return self.finish(context)
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({"INFO"}, f"正在构建 {transfer.name} · {self._log.name}")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "TIMER" and self._process.poll() is not None:
            context.window_manager.event_timer_remove(self._timer)
            return self.finish(context)
        return {"PASS_THROUGH"}

    def finish(self, context):
        if self._process.returncode:
            lines = self._log.read_text(errors="replace").strip().splitlines()
            reason = next(
                (
                    line
                    for line in reversed(lines)
                    if line.startswith(("ValueError:", "FileNotFoundError:"))
                ),
                "",
            )
            self.report({"ERROR"}, f"构建失败 {reason} · {self._log}")
            return {"CANCELLED"}
        self._scene.charloom.last_build = str(self._destination)
        if context.scene == self._scene and context.mode == "OBJECT":
            try:
                self._scene.charloom.rig, self._scene.charloom.body = (
                    api.import_character(self._destination / self._stage)
                )
            except (ValueError, RuntimeError) as error:
                self.report({"WARNING"}, f"已生成文件，导入失败 {error}")
                return {"FINISHED"}
        self.report({"INFO"}, f"已生成 {self._destination}")
        return {"FINISHED"}


class CHARLOOM_OT_import(bpy.types.Operator, ImportHelper):
    bl_idname = "charloom.import_character"
    bl_label = "打开角色 GLB"
    bl_options = {"UNDO"}
    filter_glob: StringProperty(default="*.glb", options={"HIDDEN"})

    def execute(self, context):
        try:
            settings = context.scene.charloom
            settings.rig, settings.body = api.import_character(absolute(self.filepath))
        except (OSError, ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class CHARLOOM_OT_fit(bpy.types.Operator):
    bl_idname = "charloom.fit_surface"
    bl_label = "贴合可见皮肤"
    bl_options = {"UNDO"}

    def execute(self, context):
        s = context.scene.charloom
        try:
            api.fit_surface(s.body, s.scan, s.group)
        except (ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class CHARLOOM_OT_clothing(bpy.types.Operator):
    bl_idname = "charloom.bind_clothing"
    bl_label = "将选中衣物绑定到骨架"
    bl_description = "按身体静止表面绑定衣物，让宽袖随手臂、下摆随躯干运动"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        s = context.scene.charloom
        meshes = [
            o for o in context.selected_objects if o != s.body and o.type == "MESH"
        ]
        try:
            api.bind_clothing(s.body, s.rig, meshes)
        except (ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class CHARLOOM_OT_preview(bpy.types.Operator):
    bl_idname = "charloom.preview"
    bl_label = "选择时间线动作"
    bl_options = {"UNDO"}
    clip: StringProperty()

    def execute(self, context):
        try:
            api.preview(context.scene.charloom.rig, self.clip)
        except (ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class CHARLOOM_OT_export(bpy.types.Operator, ExportHelper):
    bl_idname = "charloom.export_character"
    bl_label = "导出角色 GLB"
    filename_ext = ".glb"
    filter_glob: StringProperty(default="*.glb", options={"HIDDEN"})

    def execute(self, context):
        try:
            api.export_character(context.scene.charloom.rig, absolute(self.filepath))
        except (ValueError, RuntimeError, OSError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


class CHARLOOM_PT_character(bpy.types.Panel):
    bl_label = "Charloom"
    bl_idname = "CHARLOOM_PT_character"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Charloom"

    def draw(self, context):
        layout = self.layout
        s = context.scene.charloom
        layout.operator("charloom.import_character")
        box = layout.box()
        box.label(text="原扫描头与贴图")
        box.prop(s, "source_mode")
        if s.source_mode == "SCAN":
            for key in (
                "source_file",
                "prior",
                "guidance",
                "mouth_trace",
                "motion",
                "name",
                "segment_review",
                "segment_evidence",
            ):
                box.prop(s, key)
        else:
            box.prop(s, "transfer")
        for key in ("output", "config", "stage"):
            box.prop(s, key)
        box.operator("charloom.build")
        if s.last_build:
            box.label(text=Path(s.last_build).name, icon="FILE_FOLDER")
        layout.prop(s, "rig")
        layout.prop(s, "body")
        layout.prop(s, "scan")
        if s.body:
            layout.prop_search(s, "group", s.body, "vertex_groups")
        layout.operator("charloom.fit_surface")
        layout.operator("charloom.bind_clothing")
        row = layout.row(align=True)
        for key, label in (
            ("walking", "走路"),
            ("face_test", "表情"),
            ("face_talk", "说话"),
        ):
            row.operator("charloom.preview", text=label).clip = key
        layout.operator("charloom.export_character")


classes = (
    CharloomPreferences,
    CharloomSettings,
    CHARLOOM_OT_build,
    CHARLOOM_OT_import,
    CHARLOOM_OT_fit,
    CHARLOOM_OT_clothing,
    CHARLOOM_OT_preview,
    CHARLOOM_OT_export,
    CHARLOOM_PT_character,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.charloom = PointerProperty(type=CharloomSettings)


def unregister():
    del bpy.types.Scene.charloom
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
