class_name MiracleShip
extends RigidBody3D

@export var thruster_controller: ThrusterController
@export var control_interface: ControlInterface

var initial_global_transform: Transform3D


func _ready() -> void:
	initial_global_transform = global_transform

	if thruster_controller == null:
		push_warning("MiracleShip is missing its ThrusterController reference")
	if control_interface == null:
		push_warning("MiracleShip is missing its ControlInterface reference")


func get_thruster_controller() -> ThrusterController:
	return thruster_controller


func get_control_interface() -> ControlInterface:
	return control_interface


func get_linear_velocity_readout() -> Vector3:
	return linear_velocity


func get_angular_velocity_readout() -> Vector3:
	return angular_velocity


func get_local_vector(world_vector: Vector3) -> Vector3:
	return global_transform.basis.inverse() * world_vector


func reset_state() -> void:
	global_transform = initial_global_transform
	linear_velocity = Vector3.ZERO
	angular_velocity = Vector3.ZERO
	constant_force = Vector3.ZERO
	constant_torque = Vector3.ZERO
	sleeping = false

	if thruster_controller != null:
		thruster_controller.stop_all()