class_name GoalArea
extends Area3D

signal occupancy_changed(is_inside: bool)
signal completion_changed(is_completed: bool)
signal goal_completed(relative_speed: float)
signal relative_speed_changed(relative_speed: float)

@export var ship: MiracleShip
@export_range(0.0, 1000.0, 0.1, "or_greater") var speed_threshold_mps := 2.5
@export var goal_velocity: Vector3 = Vector3.ZERO

var ship_inside := false
var is_completed := false
var current_relative_speed := 0.0


func _ready() -> void:
	monitoring = true
	body_entered.connect(_on_body_entered)
	body_exited.connect(_on_body_exited)
	if ship == null:
		push_warning("GoalArea is missing its ship reference")
	if speed_threshold_mps < 0.0:
		push_warning("GoalArea speed threshold should not be negative")


func _physics_process(_delta: float) -> void:
	current_relative_speed = get_relative_speed()
	relative_speed_changed.emit(current_relative_speed)
	if ship_inside and not is_completed and current_relative_speed <= speed_threshold_mps:
		is_completed = true
		completion_changed.emit(true)
		goal_completed.emit(current_relative_speed)


func get_relative_velocity() -> Vector3:
	if ship == null:
		return Vector3.ZERO

	return ship.get_linear_velocity_readout() - goal_velocity


func get_relative_speed() -> float:
	return get_relative_velocity().length()


func is_goal_completed() -> bool:
	return is_completed


func is_ship_inside() -> bool:
	return ship_inside


func reset_goal() -> void:
	if not is_completed:
		return

	is_completed = false
	completion_changed.emit(false)


func _on_body_entered(body: Node) -> void:
	if body != ship:
		return

	ship_inside = true
	occupancy_changed.emit(true)


func _on_body_exited(body: Node) -> void:
	if body != ship:
		return

	ship_inside = false
	occupancy_changed.emit(false)