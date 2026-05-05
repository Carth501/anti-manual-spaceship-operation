class_name UIManager
extends Control

@export var control_interface_path: NodePath = ^"../Miracle/Node"
@export var throttle_slider_path: NodePath = ^"VSlider"

var control_interface: ControlInterface
var throttle_slider: VSlider


func _ready() -> void:
	control_interface = get_node_or_null(control_interface_path) as ControlInterface
	throttle_slider = get_node_or_null(throttle_slider_path) as VSlider

	if control_interface == null:
		push_warning("UIManager could not find a ControlInterface at %s" % control_interface_path)
		return

	if throttle_slider == null:
		push_warning("UIManager could not find a VSlider at %s" % throttle_slider_path)
		return

	_configure_throttle_slider()
	throttle_slider.value_changed.connect(_on_throttle_slider_changed)
	control_interface.main_throttle_changed.connect(_on_main_throttle_changed)
	_sync_slider_to_throttle()


func _configure_throttle_slider() -> void:
	throttle_slider.min_value = 0.0
	throttle_slider.max_value = 1.0
	throttle_slider.step = 0.01


func _sync_slider_to_throttle() -> void:
	throttle_slider.value = control_interface.get_main_throttle()


func _on_throttle_slider_changed(value: float) -> void:
	control_interface.set_main_throttle(value)


func _on_main_throttle_changed(value: float) -> void:
	throttle_slider.value = value