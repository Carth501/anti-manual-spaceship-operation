class_name ControlInterface
extends Node

signal main_throttle_changed(value: float)
signal control_axes_changed(values: Dictionary)

enum InputMode {
	PLAYER,
	RL_DIRECT_THRUSTERS,
}

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
var input_mode: InputMode = InputMode.PLAYER
var rl_thruster_inputs: Array[float] = []


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

	if input_mode == InputMode.RL_DIRECT_THRUSTERS:
		_set_control_axes(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
		thruster_controller.set_direct_throttles(rl_thruster_inputs)
	else:
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


func set_rl_control_enabled(enabled: bool) -> void:
	input_mode = InputMode.RL_DIRECT_THRUSTERS if enabled else InputMode.PLAYER
	if enabled:
		_set_control_axes(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
		return

	rl_thruster_inputs.clear()
	thruster_controller.stop_all()


func is_rl_control_enabled() -> bool:
	return input_mode == InputMode.RL_DIRECT_THRUSTERS


func set_rl_thruster_inputs(throttle_values: Array[float]) -> void:
	input_mode = InputMode.RL_DIRECT_THRUSTERS
	rl_thruster_inputs = throttle_values.duplicate()


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
	_set_control_axes(
		Input.get_action_strength("Lateral_Right") - Input.get_action_strength("Lateral_Left"),
		Input.get_action_strength("Lateral_Up") - Input.get_action_strength("Lateral_Down"),
		main_throttle + Input.get_action_strength("Lateral_Forward") - Input.get_action_strength("Lateral_Back"),
		Input.get_action_strength("Pitch_Up") - Input.get_action_strength("Pitch_Down"),
		Input.get_action_strength("Yaw_Left") - Input.get_action_strength("Yaw_Right"),
		Input.get_action_strength("Roll_Left") - Input.get_action_strength("Roll_Right")
	)


func _set_control_axes(
		horizontal: float,
		vertical: float,
		fore: float,
		pitch: float,
		yaw: float,
		roll: float
	) -> void:
	if is_equal_approx(current_horizontal_input, horizontal) \
			and is_equal_approx(current_vertical_input, vertical) \
			and is_equal_approx(current_fore_input, fore) \
			and is_equal_approx(current_pitch_input, pitch) \
			and is_equal_approx(current_yaw_input, yaw) \
			and is_equal_approx(current_roll_input, roll):
		return

	current_horizontal_input = horizontal
	current_vertical_input = vertical
	current_fore_input = fore
	current_pitch_input = pitch
	current_yaw_input = yaw
	current_roll_input = roll
	control_axes_changed.emit(get_control_axes())


func _normalized_axis(axis: Vector3) -> Vector3:
	if axis.is_zero_approx():
		return Vector3.ZERO
	return axis.normalized()
