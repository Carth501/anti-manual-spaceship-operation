class_name ControlInterface
extends Node

signal main_throttle_changed(value: float)

@export var thruster_controller_path: NodePath = ^"../ThrustController"
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


func _ready() -> void:
	thruster_controller = get_node_or_null(thruster_controller_path) as ThrusterController
	if thruster_controller == null:
		push_warning("ControlInterface could not find a ThrusterController at %s" % thruster_controller_path)


func _physics_process(delta: float) -> void:
	if thruster_controller == null:
		return

	_update_main_throttle(delta)
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


func _build_linear_request() -> Vector3:
	var lateral_x := Input.get_action_strength("Lateral_Right") - Input.get_action_strength("Lateral_Left")
	var lateral_y := Input.get_action_strength("Lateral_Up") - Input.get_action_strength("Lateral_Down")
	var forward_request := main_throttle
	forward_request += Input.get_action_strength("Lateral_Forward")
	forward_request -= Input.get_action_strength("Lateral_Back")

	var linear_request := _normalized_axis(local_right_axis) * lateral_x
	linear_request += _normalized_axis(local_up_axis) * lateral_y
	linear_request += _normalized_axis(local_forward_axis) * forward_request
	return linear_request.limit_length(1.0) * lateral_authority


func _build_angular_request() -> Vector3:
	var pitch_request := Input.get_action_strength("Pitch_Up") - Input.get_action_strength("Pitch_Down")
	var yaw_request := Input.get_action_strength("Yaw_Left") - Input.get_action_strength("Yaw_Right")
	var roll_request := Input.get_action_strength("Roll_Left") - Input.get_action_strength("Roll_Right")

	var angular_request := _normalized_axis(local_pitch_axis) * pitch_request
	angular_request += _normalized_axis(local_yaw_axis) * yaw_request
	angular_request += _normalized_axis(local_roll_axis) * roll_request
	return angular_request.limit_length(1.0) * rotational_authority


func _normalized_axis(axis: Vector3) -> Vector3:
	if axis.is_zero_approx():
		return Vector3.ZERO
	return axis.normalized()
