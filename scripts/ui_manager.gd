class_name UIManager
extends Control

@export var control_interface_path: NodePath = ^"../Miracle/Node"
@export var throttle_slider_path: NodePath = ^"VSlider"
@export var throttle_label_path: NodePath = ^"Control/Label"

var control_interface: ControlInterface
var throttle_slider: VSlider
var throttle_label: Label


func _ready() -> void:
	control_interface = get_node_or_null(control_interface_path) as ControlInterface
	throttle_slider = get_node_or_null(throttle_slider_path) as VSlider
	throttle_label = get_node_or_null(throttle_label_path) as Label

	if control_interface == null:
		push_warning("UIManager could not find a ControlInterface at %s" % control_interface_path)
		return

	if throttle_slider == null:
		push_warning("UIManager could not find a VSlider at %s" % throttle_slider_path)

	if throttle_label == null:
		push_warning("UIManager could not find a Label at %s" % throttle_label_path)

	if throttle_slider != null:
		_configure_throttle_slider()
		throttle_slider.value_changed.connect(_on_throttle_slider_changed)
	control_interface.main_throttle_changed.connect(_on_main_throttle_changed)
	_sync_ui_to_throttle()


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