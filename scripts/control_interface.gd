class_name ControlInterface
extends Node

signal main_throttle_changed(value: float)
signal control_axes_changed(values: Dictionary)

@export var ship: MiracleShip
@export var local_right_axis: Vector3 = Vector3.RIGHT
@export var local_up_axis: Vector3 = Vector3.BACK
@export var local_forward_axis: Vector3 = Vector3.UP
@export var local_pitch_axis: Vector3 = Vector3.RIGHT
@export var local_yaw_axis: Vector3 = Vector3.BACK
@export var local_roll_axis: Vector3 = Vector3.UP
@export_range(0.0, 1.0, 0.01) var main_throttle: float = 0.0
@export_range(0.0, 1.0, 0.01) var main_throttle_change_rate: float = 1
@export_range(0.0, 1.0, 0.01) var lateral_authority: float = 1.0
@export_range(0.0, 1.0, 0.01) var rotational_authority: float = 1.0

var thruster_controller: ThrusterController
var current_horizontal_input: float = 0.0
var current_vertical_input: float = 0.0
var current_fore_input: float = 0.0
var current_pitch_input: float = 0.0
var current_yaw_input: float = 0.0
var current_roll_input: float = 0.0


func _ready() -> void:
	if ship == null:
		ship = get_parent() as MiracleShip
	if ship == null:
		push_warning("ControlInterface could not find a MiracleShip reference")
		return

	thruster_controller = ship.get_thruster_controller()
	if thruster_controller == null:
		push_warning("ControlInterface could not resolve a ThrusterController from its ship")


func _physics_process(delta: float) -> void:
	if thruster_controller == null:
		return

	_update_main_throttle(delta)
	_update_control_axes()
	thruster_controller.set_input(_build_linear_request(), _build_angular_request())
	thruster_controller.apply_current_forces()


func _update_main_throttle(delta: float) -> void:
	var throttle_change := Input.get_action_strength("Increase_Main_Thruster")
	throttle_change -= Input.get_action_strength("Decrease_Main_Thruster")
	set_main_throttle(main_throttle + (throttle_change * main_throttle_change_rate * delta))


func set_main_throttle(value: float) -> void:
	var clamped_value: float = clamp(value, 0.0, 1.0)
	if is_equal_approx(main_throttle, clamped_value):
		return

	main_throttle = clamped_value
	main_throttle_changed.emit(main_throttle)


func get_main_throttle() -> float:
	return main_throttle


func get_control_axes() -> Dictionary:
	return {
		"horizontal": current_horizontal_input,
		"vertical": current_vertical_input,
		"fore": current_fore_input,
		"pitch": current_pitch_input,
		"yaw": current_yaw_input,
		"roll": current_roll_input,
	}


func _build_linear_request() -> Vector3:
	var linear_request := _normalized_axis(local_right_axis) * current_horizontal_input
	linear_request += _normalized_axis(local_up_axis) * current_vertical_input
	linear_request += _normalized_axis(local_forward_axis) * current_fore_input
	return linear_request.limit_length(1.0) * lateral_authority


func _build_angular_request() -> Vector3:
	var angular_request := _normalized_axis(local_pitch_axis) * current_pitch_input
	angular_request += _normalized_axis(local_yaw_axis) * current_yaw_input
	angular_request += _normalized_axis(local_roll_axis) * current_roll_input
	return angular_request.limit_length(1.0) * rotational_authority


func _update_control_axes() -> void:
	current_horizontal_input = Input.get_action_strength("Lateral_Right") - Input.get_action_strength("Lateral_Left")
	current_vertical_input = Input.get_action_strength("Lateral_Up") - Input.get_action_strength("Lateral_Down")
	current_fore_input = main_throttle
	current_fore_input += Input.get_action_strength("Lateral_Forward")
	current_fore_input -= Input.get_action_strength("Lateral_Back")
	current_pitch_input = Input.get_action_strength("Pitch_Up") - Input.get_action_strength("Pitch_Down")
	current_yaw_input = Input.get_action_strength("Yaw_Left") - Input.get_action_strength("Yaw_Right")
	current_roll_input = Input.get_action_strength("Roll_Left") - Input.get_action_strength("Roll_Right")
	control_axes_changed.emit(get_control_axes())


func _normalized_axis(axis: Vector3) -> Vector3:
	if axis.is_zero_approx():
		return Vector3.ZERO
	return axis.normalized()
