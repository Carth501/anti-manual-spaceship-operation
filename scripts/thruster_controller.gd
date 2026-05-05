class_name ThrusterController
extends Node3D

@export var center_of_mass_local: Vector3 = Vector3.ZERO
@export var rigid_body_path: NodePath = ^".."
@export var thruster_paths: Array[NodePath] = []

var current_linear_request: Vector3 = Vector3.ZERO
var current_angular_request: Vector3 = Vector3.ZERO
var direct_thruster_mode_enabled := false
var rigid_body: RigidBody3D
var thrusters: Array[ThrusterPoint] = []


func _ready() -> void:
	rigid_body = get_node_or_null(rigid_body_path) as RigidBody3D
	if rigid_body == null:
		push_warning("ThrusterController could not find a RigidBody3D at %s" % rigid_body_path)
	refresh_thrusters()


func refresh_thrusters() -> void:
	thrusters.clear()

	if not thruster_paths.is_empty():
		for thruster_path in thruster_paths:
			var path_thruster := get_node_or_null(thruster_path) as ThrusterPoint
			if path_thruster != null:
				thrusters.append(path_thruster)
	else:
		for node in find_children("*", "", true, false):
			var found_thruster := node as ThrusterPoint
			if found_thruster != null:
				thrusters.append(found_thruster)

	_update_thrusters()


func set_input(local_linear_request: Vector3, local_angular_request: Vector3) -> void:
	direct_thruster_mode_enabled = false
	current_linear_request = local_linear_request
	current_angular_request = local_angular_request
	_update_thrusters()


func set_direct_throttles(throttle_values: Array[float]) -> void:
	direct_thruster_mode_enabled = true
	current_linear_request = Vector3.ZERO
	current_angular_request = Vector3.ZERO

	for index in thrusters.size():
		var target_throttle := 0.0
		if index < throttle_values.size():
			target_throttle = throttle_values[index]
		thrusters[index].set_throttle(target_throttle)


func get_thruster_count() -> int:
	return thrusters.size()


func get_current_throttles() -> Array[float]:
	var throttle_values: Array[float] = []
	throttle_values.resize(thrusters.size())
	for index in thrusters.size():
		throttle_values[index] = thrusters[index].current_throttle
	return throttle_values


func apply_current_forces() -> void:
	if rigid_body == null:
		return

	var total_force_local := get_total_force_local()
	if not total_force_local.is_zero_approx():
		rigid_body.apply_central_force(global_transform.basis * total_force_local)

	var total_torque_local := get_total_torque_local()
	if not total_torque_local.is_zero_approx():
		rigid_body.apply_torque(global_transform.basis * total_torque_local)


func stop_all() -> void:
	if direct_thruster_mode_enabled:
		set_direct_throttles([])
		return

	set_input(Vector3.ZERO, Vector3.ZERO)


func get_total_force_local() -> Vector3:
	var total_force := Vector3.ZERO
	for thruster in thrusters:
		total_force += thruster.get_force_vector_local(self )
	return total_force


func get_total_torque_local() -> Vector3:
	var total_torque := Vector3.ZERO
	for thruster in thrusters:
		total_torque += thruster.get_torque_vector_local(self , center_of_mass_local)
	return total_torque


func _update_thrusters() -> void:
	var max_torque_leverage := _get_max_torque_leverage()
	for thruster in thrusters:
		thruster.apply_command(
			self ,
			current_linear_request,
			current_angular_request,
			center_of_mass_local,
			max_torque_leverage
		)


func _get_max_torque_leverage() -> float:
	var max_torque := 0.0
	for thruster in thrusters:
		max_torque = max(max_torque, thruster.get_unit_torque_vector_local(self , center_of_mass_local).length())
	return max_torque