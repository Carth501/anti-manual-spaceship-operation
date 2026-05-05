class_name MiracleShip
extends RigidBody3D

@export var thruster_controller: ThrusterController
@export var control_interface: ControlInterface


func _ready() -> void:
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