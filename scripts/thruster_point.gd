class_name ThrusterPoint
extends Node3D

@export var thrust_direction: Vector3 = Vector3.UP
@export_range(0.0, 1.0, 0.01) var linear_response: float = 1.0
@export_range(0.0, 1.0, 0.01) var angular_response: float = 1.0
@export_range(0.0, 1000.0, 0.1, "or_greater") var max_force: float = 1.0
@export var enabled: bool = true
@export var feedback_particles_path: NodePath = ^"GPUParticles3D"
@export_range(0.0, 1.0, 0.01) var feedback_deadzone: float = 0.01
@export_range(0.0, 4.0, 0.01, "or_greater") var feedback_idle_speed_scale: float = 0.25
@export_range(0.0, 4.0, 0.01, "or_greater") var feedback_full_speed_scale: float = 1.0

var current_throttle: float = 0.0
var feedback_particles: GPUParticles3D


func _ready() -> void:
	feedback_particles = get_node_or_null(feedback_particles_path) as GPUParticles3D
	_update_feedback()


func apply_command(
		reference_frame: Node3D,
		local_linear_request: Vector3,
		local_angular_request: Vector3,
		center_of_mass_local: Vector3,
		max_torque_leverage: float
	) -> void:
	current_throttle = evaluate_command(
		reference_frame,
		local_linear_request,
		local_angular_request,
		center_of_mass_local,
		max_torque_leverage
	)
	_update_feedback()


func evaluate_command(
		reference_frame: Node3D,
		local_linear_request: Vector3,
		local_angular_request: Vector3,
		center_of_mass_local: Vector3,
		max_torque_leverage: float
	) -> float:
	if not enabled or max_force <= 0.0:
		return 0.0

	var force_direction := get_force_direction_local(reference_frame)
	if force_direction.is_zero_approx():
		return 0.0

	var linear_amount := local_linear_request.length()
	var linear_score := 0.0
	if linear_amount > 0.0:
		linear_score = max(force_direction.dot(local_linear_request / linear_amount), 0.0) * linear_amount

	var unit_torque_vector := get_unit_torque_vector_local(reference_frame, center_of_mass_local)
	var angular_amount := local_angular_request.length()
	var angular_score := 0.0
	if angular_amount > 0.0 and not unit_torque_vector.is_zero_approx():
		var leverage_scale := 1.0
		if max_torque_leverage > 0.0:
			leverage_scale = clamp(unit_torque_vector.length() / max_torque_leverage, 0.0, 1.0)
		angular_score = max(unit_torque_vector.normalized().dot(local_angular_request / angular_amount), 0.0)
		angular_score *= angular_amount * leverage_scale

	return clamp((linear_score * linear_response) + (angular_score * angular_response), 0.0, 1.0)


func get_force_vector_local(reference_frame: Node3D) -> Vector3:
	return get_force_direction_local(reference_frame) * current_throttle * max_force


func get_torque_vector_local(reference_frame: Node3D, center_of_mass_local: Vector3) -> Vector3:
	var lever_arm := reference_frame.to_local(global_position) - center_of_mass_local
	return lever_arm.cross(get_force_vector_local(reference_frame))


func get_force_direction_local(reference_frame: Node3D) -> Vector3:
	var local_direction := thrust_direction.normalized()
	if local_direction.is_zero_approx():
		return Vector3.ZERO

	var world_direction := global_transform.basis * local_direction
	return (reference_frame.global_transform.basis.inverse() * world_direction).normalized()


func get_unit_torque_vector_local(reference_frame: Node3D, center_of_mass_local: Vector3) -> Vector3:
	var lever_arm := reference_frame.to_local(global_position) - center_of_mass_local
	return lever_arm.cross(get_force_direction_local(reference_frame))


func _update_feedback() -> void:
	if feedback_particles == null:
		return

	var feedback_amount: float = clamp(current_throttle, 0.0, 1.0)
	var is_firing := enabled and feedback_amount > feedback_deadzone
	feedback_particles.emitting = is_firing
	feedback_particles.amount_ratio = feedback_amount
	feedback_particles.speed_scale = lerpf(feedback_idle_speed_scale, feedback_full_speed_scale, feedback_amount)
