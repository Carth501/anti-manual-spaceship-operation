class_name UIManager
extends Control

@export var ship: MiracleShip
@export var throttle_slider_path: NodePath = ^"VSlider"
@export var throttle_label_path: NodePath = ^"Control/Label"
@export var roll_value_label_path: NodePath = ^"HBoxContainer/VBoxContainer/Roll/Value"
@export var pitch_value_label_path: NodePath = ^"HBoxContainer/VBoxContainer/Pitch/Value"
@export var yaw_value_label_path: NodePath = ^"HBoxContainer/VBoxContainer/Yaw/Value"
@export var horizontal_value_label_path: NodePath = ^"HBoxContainer/VBoxContainer/Horizontal/Value"
@export var vertical_value_label_path: NodePath = ^"HBoxContainer/VBoxContainer/Verticle/Value"
@export var fore_value_label_path: NodePath = ^"HBoxContainer/VBoxContainer/Fore/Value"
@export var linear_x_value_label_path: NodePath = ^"Velocities/Linear/XRow/Value"
@export var linear_y_value_label_path: NodePath = ^"Velocities/Linear/YRow/Value"
@export var linear_z_value_label_path: NodePath = ^"Velocities/Linear/ZRow/Value"
@export var linear_total_value_label_path: NodePath = ^"Velocities/Linear/TotalRow/Value"
@export var angular_x_value_label_path: NodePath = ^"Velocities/Angular/XRow/Value"
@export var angular_y_value_label_path: NodePath = ^"Velocities/Angular/YRow/Value"
@export var angular_z_value_label_path: NodePath = ^"Velocities/Angular/ZRow/Value"
@export var angular_total_value_label_path: NodePath = ^"Velocities/Angular/TotalRow/Value"

var control_interface: ControlInterface
var throttle_slider: VSlider
var throttle_label: Label
var roll_value_label: Label
var pitch_value_label: Label
var yaw_value_label: Label
var horizontal_value_label: Label
var vertical_value_label: Label
var fore_value_label: Label
var linear_x_value_label: Label
var linear_y_value_label: Label
var linear_z_value_label: Label
var linear_total_value_label: Label
var angular_x_value_label: Label
var angular_y_value_label: Label
var angular_z_value_label: Label
var angular_total_value_label: Label


func _ready() -> void:
	if ship == null:
		push_warning("UIManager is missing its ship reference")
		return

	control_interface = ship.get_control_interface()
	throttle_slider = get_node_or_null(throttle_slider_path) as VSlider
	throttle_label = get_node_or_null(throttle_label_path) as Label
	roll_value_label = get_node_or_null(roll_value_label_path) as Label
	pitch_value_label = get_node_or_null(pitch_value_label_path) as Label
	yaw_value_label = get_node_or_null(yaw_value_label_path) as Label
	horizontal_value_label = get_node_or_null(horizontal_value_label_path) as Label
	vertical_value_label = get_node_or_null(vertical_value_label_path) as Label
	fore_value_label = get_node_or_null(fore_value_label_path) as Label
	linear_x_value_label = get_node_or_null(linear_x_value_label_path) as Label
	linear_y_value_label = get_node_or_null(linear_y_value_label_path) as Label
	linear_z_value_label = get_node_or_null(linear_z_value_label_path) as Label
	linear_total_value_label = get_node_or_null(linear_total_value_label_path) as Label
	angular_x_value_label = get_node_or_null(angular_x_value_label_path) as Label
	angular_y_value_label = get_node_or_null(angular_y_value_label_path) as Label
	angular_z_value_label = get_node_or_null(angular_z_value_label_path) as Label
	angular_total_value_label = get_node_or_null(angular_total_value_label_path) as Label

	if control_interface == null:
		push_warning("UIManager could not resolve a ControlInterface from its ship")
		return

	if throttle_slider == null:
		push_warning("UIManager could not find a VSlider at %s" % throttle_slider_path)

	if throttle_label == null:
		push_warning("UIManager could not find a Label at %s" % throttle_label_path)

	if throttle_slider != null:
		_configure_throttle_slider()
		throttle_slider.value_changed.connect(_on_throttle_slider_changed)
	control_interface.main_throttle_changed.connect(_on_main_throttle_changed)
	control_interface.control_axes_changed.connect(_on_control_axes_changed)
	_sync_ui_to_throttle()
	_update_axis_labels(control_interface.get_control_axes())
	_update_momentum_labels()


func _physics_process(_delta: float) -> void:
	if ship == null:
		return

	_update_momentum_labels()


func _configure_throttle_slider() -> void:
	throttle_slider.min_value = 0.0
	throttle_slider.max_value = 1.0
	throttle_slider.step = 0.01


func _sync_ui_to_throttle() -> void:
	var throttle_value := control_interface.get_main_throttle()
	if throttle_slider != null:
		throttle_slider.value = throttle_value
	_update_throttle_label(throttle_value)


func _on_throttle_slider_changed(value: float) -> void:
	control_interface.set_main_throttle(value)


func _on_main_throttle_changed(value: float) -> void:
	if throttle_slider != null:
		throttle_slider.value = value
	_update_throttle_label(value)


func _update_throttle_label(value: float) -> void:
	if throttle_label == null:
		return

	throttle_label.text = "%d%%" % int(round(value * 100.0))


func _on_control_axes_changed(values: Dictionary) -> void:
	_update_axis_labels(values)


func _update_axis_labels(values: Dictionary) -> void:
	_update_axis_label(horizontal_value_label, values.get("horizontal", 0.0))
	_update_axis_label(vertical_value_label, values.get("vertical", 0.0))
	_update_axis_label(fore_value_label, values.get("fore", 0.0))
	_update_axis_label(pitch_value_label, values.get("pitch", 0.0))
	_update_axis_label(yaw_value_label, values.get("yaw", 0.0))
	_update_axis_label(roll_value_label, values.get("roll", 0.0))


func _update_axis_label(target_label: Label, value: float) -> void:
	if target_label == null:
		return

	target_label.text = "%+d%%" % int(round(value * 100.0))


func _update_momentum_labels() -> void:
	var linear_momentum := ship.get_linear_momentum()
	var angular_momentum := ship.get_angular_momentum()

	_update_signed_value_label(linear_x_value_label, linear_momentum.x)
	_update_signed_value_label(linear_y_value_label, linear_momentum.y)
	_update_signed_value_label(linear_z_value_label, linear_momentum.z)
	_update_unsigned_value_label(linear_total_value_label, linear_momentum.length())
 
	_update_signed_value_label(angular_x_value_label, angular_momentum.x)
	_update_signed_value_label(angular_y_value_label, angular_momentum.y)
	_update_signed_value_label(angular_z_value_label, angular_momentum.z)
	_update_unsigned_value_label(angular_total_value_label, angular_momentum.length())


func _update_signed_value_label(target_label: Label, value: float) -> void:
	if target_label == null:
		return

	target_label.text = "%+.1f" % value


func _update_unsigned_value_label(target_label: Label, value: float) -> void:
	if target_label == null:
		return

	target_label.text = "%.1f" % value