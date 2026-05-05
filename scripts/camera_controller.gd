class_name CameraController
extends Node3D

signal orbit_button_state_changed(is_pressed: bool, status_text: String)

@export var camera_path: NodePath = ^"Camera3D"
@export var orbit_action: StringName = &"Camera_Control"
@export_range(0.01, 1.0, 0.01) var mouse_sensitivity_degrees: float = 0.15
@export_range(-89.0, 0.0, 0.1) var min_pitch_degrees: float = -80.0
@export_range(0.0, 89.0, 0.1) var max_pitch_degrees: float = 80.0

var is_dragging: bool = false
var yaw_degrees: float = 0.0
var pitch_degrees: float = 0.0

@onready var orbit_camera: Camera3D = get_node_or_null(camera_path) as Camera3D


func _ready() -> void:
	yaw_degrees = rotation_degrees.y
	pitch_degrees = clamp(rotation_degrees.x, min_pitch_degrees, max_pitch_degrees)
	_apply_orbit_rotation()
	_emit_orbit_button_status()

	if orbit_camera == null:
		push_warning("CameraController could not find its Camera3D child")

	if not InputMap.has_action(orbit_action):
		push_warning("CameraController could not find the %s input action" % String(orbit_action))


func _input(event: InputEvent) -> void:
	if event.is_action_pressed(orbit_action):
		is_dragging = true
		_emit_orbit_button_status()
		return

	if event.is_action_released(orbit_action):
		is_dragging = false
		_emit_orbit_button_status()
		return

	if not is_dragging:
		return

	if event is InputEventMouseMotion:
		yaw_degrees = wrapf(yaw_degrees - (event.relative.x * mouse_sensitivity_degrees), -180.0, 180.0)
		pitch_degrees = clamp(
			pitch_degrees + (event.relative.y * mouse_sensitivity_degrees),
			min_pitch_degrees,
			max_pitch_degrees
		)
		_apply_orbit_rotation()


func _apply_orbit_rotation() -> void:
	rotation_degrees = Vector3(pitch_degrees, yaw_degrees, 0.0)


func get_orbit_button_status_text() -> String:
	var orbit_state := "pressed" if is_dragging else "released"
	return "%s: %s" % [String(orbit_action), orbit_state]


func _emit_orbit_button_status() -> void:
	orbit_button_state_changed.emit(is_dragging, get_orbit_button_status_text())
