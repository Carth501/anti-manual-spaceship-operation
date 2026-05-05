class_name RLBridge
extends Node

@export var ship: MiracleShip
@export var goal_area: GoalArea
@export var auto_start_server := true
@export var listen_host := "127.0.0.1"
@export_range(1024, 65535, 1) var listen_port := 8765
@export_range(1, 60, 1) var default_action_frames := 4
@export_range(1, 100000, 1) var episode_frame_limit := 2400
@export_range(0.0, 100000.0, 0.1, "or_greater") var training_boundary_radius := 1500.0
@export_range(0.0, 100.0, 0.001, "or_greater") var progress_reward_scale := 0.05
@export_range(0.0, 100.0, 0.001, "or_greater") var speed_penalty_scale := 0.02
@export_range(0.0, 100.0, 0.001, "or_greater") var thruster_penalty_scale := 0.01
@export_range(0.0, 10.0, 0.0001, "or_greater") var living_penalty_per_frame := 0.001
@export_range(0.0, 10000.0, 0.1, "or_greater") var success_reward := 100.0
@export_range(0.0, 10000.0, 0.1, "or_greater") var out_of_bounds_penalty := 25.0

var server := TCPServer.new()
var client: StreamPeerTCP
var control_interface: ControlInterface
var receive_buffer := ""
var episode_frames := 0
var previous_goal_distance := 0.0
var pending_step_frames := 0
var pending_action_frames := 0
var spawn_origin := Vector3.ZERO


func _ready() -> void:
	if ship == null:
		push_warning("RLBridge is missing its ship reference")
		return

	control_interface = ship.get_control_interface()
	if control_interface == null:
		push_warning("RLBridge could not resolve a ControlInterface from its ship")
		return

	if goal_area == null:
		push_warning("RLBridge is missing its goal area reference")
		return

	_reset_episode_state()
	spawn_origin = ship.global_position
	if auto_start_server:
		_start_server()


func _exit_tree() -> void:
	_release_rl_control()
	if client != null:
		client.disconnect_from_host()
	if server.is_listening():
		server.stop()


func _process(_delta: float) -> void:
	_poll_server()
	if pending_step_frames == 0:
		_poll_client_messages()


func _physics_process(_delta: float) -> void:
	if pending_step_frames <= 0:
		return

	episode_frames += 1
	pending_step_frames -= 1
	if pending_step_frames == 0:
		_complete_step()


func _start_server() -> void:
	if server.is_listening():
		return

	var listen_error := server.listen(listen_port, listen_host)
	if listen_error != OK:
		push_warning("RLBridge failed to listen on %s:%d (error %d)" % [listen_host, listen_port, listen_error])


func _poll_server() -> void:
	if not server.is_listening():
		return

	if client != null and client.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		client.poll()
		return

	if server.is_connection_available():
		client = server.take_connection()
		receive_buffer = ""


func _poll_client_messages() -> void:
	if client == null:
		return

	client.poll()
	if client.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		_release_rl_control()
		client = null
		receive_buffer = ""
		return

	var available_bytes := client.get_available_bytes()
	if available_bytes <= 0:
		return

	receive_buffer += client.get_utf8_string(available_bytes)
	var separator_index := receive_buffer.find("\n")
	while separator_index >= 0:
		var message := receive_buffer.substr(0, separator_index).strip_edges()
		receive_buffer = receive_buffer.substr(separator_index + 1)
		if not message.is_empty():
			_handle_message(message)
		separator_index = receive_buffer.find("\n")


func _handle_message(message: String) -> void:
	var parsed_message = JSON.parse_string(message)
	if typeof(parsed_message) != TYPE_DICTIONARY:
		_send_error("invalid_json", "Expected a JSON object command")
		return

	var command := String(parsed_message.get("command", ""))
	match command:
		"hello":
			_send_response(_build_hello_response())
		"reset":
			_reset_episode_state()
			_send_response(_build_step_response(0.0, false, "reset"))
		"step":
			_begin_step(parsed_message)
		"close":
			_send_response({"ok": true})
			_release_rl_control()
			client.disconnect_from_host()
			client = null
			receive_buffer = ""
		_:
			_send_error("unknown_command", "Unknown command: %s" % command)


func _begin_step(command: Dictionary) -> void:
	if pending_step_frames > 0:
		_send_error("busy", "A step is already in progress")
		return

	var action_values := _coerce_float_array(command.get("action", []))
	control_interface.set_rl_control_enabled(true)
	control_interface.set_rl_thruster_inputs(action_values)
	pending_action_frames = max(int(command.get("frames", default_action_frames)), 1)
	pending_step_frames = pending_action_frames


func _complete_step() -> void:
	var reward_terms := _compute_reward_terms()
	var total_reward := reward_terms.get("total", 0.0)
	var done := false
	var terminal_reason := "in_progress"

	if goal_area.is_goal_completed():
		total_reward += success_reward
		done = true
		terminal_reason = "goal_reached"
	elif _is_out_of_bounds():
		total_reward -= out_of_bounds_penalty
		done = true
		terminal_reason = "out_of_bounds"
	elif episode_frames >= episode_frame_limit:
		done = true
		terminal_reason = "timeout"

	previous_goal_distance = _get_goal_distance()
	_send_response(_build_step_response(total_reward, done, terminal_reason, reward_terms))


func _reset_episode_state() -> void:
	control_interface.set_rl_control_enabled(true)
	control_interface.set_rl_thruster_inputs([])
	ship.reset_state()
	goal_area.reset_goal()
	goal_area.refresh_goal_state()
	episode_frames = 0
	previous_goal_distance = _get_goal_distance()
	pending_step_frames = 0
	pending_action_frames = 0


func _release_rl_control() -> void:
	if control_interface == null:
		return

	control_interface.set_rl_control_enabled(false)


func _build_hello_response() -> Dictionary:
	return {
		"ok": true,
		"version": 1,
		"thruster_count": ship.get_thruster_controller().get_thruster_count(),
		"default_action_frames": default_action_frames,
		"observation": _get_observation(),
	}


func _build_step_response(
		reward: float,
		done: bool,
		terminal_reason: String,
		reward_terms: Dictionary = {}
	) -> Dictionary:
	return {
		"ok": true,
		"observation": _get_observation(),
		"reward": reward,
		"done": done,
		"info": {
			"terminal_reason": terminal_reason,
			"episode_frames": episode_frames,
			"goal_distance": _get_goal_distance(),
			"relative_speed": goal_area.get_relative_speed(),
			"is_inside_goal": goal_area.is_ship_inside(),
			"is_goal_completed": goal_area.is_goal_completed(),
			"reward_terms": reward_terms,
		},
	}


func _compute_reward_terms() -> Dictionary:
	var current_goal_distance := _get_goal_distance()
	var progress_reward := (previous_goal_distance - current_goal_distance) * progress_reward_scale
	var speed_penalty := goal_area.get_relative_speed() * speed_penalty_scale
	var throttles := ship.get_thruster_controller().get_current_throttles()
	var throttle_sum := 0.0
	for throttle_value in throttles:
		throttle_sum += throttle_value
	var thruster_penalty := throttle_sum * thruster_penalty_scale
	var living_penalty := float(max(pending_action_frames, 1)) * living_penalty_per_frame
	var total_reward := progress_reward - speed_penalty - thruster_penalty - living_penalty

	return {
		"progress": progress_reward,
		"speed_penalty": speed_penalty,
		"thruster_penalty": thruster_penalty,
		"living_penalty": living_penalty,
		"total": total_reward,
	}


func _get_observation() -> Array[float]:
	var goal_offset_local := ship.get_local_vector(goal_area.global_position - ship.global_position)
	var linear_velocity_local := ship.get_local_vector(ship.get_linear_velocity_readout())
	var angular_velocity_local := ship.get_local_vector(ship.get_angular_velocity_readout())
	var observation: Array[float] = [
		goal_offset_local.x,
		goal_offset_local.y,
		goal_offset_local.z,
		linear_velocity_local.x,
		linear_velocity_local.y,
		linear_velocity_local.z,
		angular_velocity_local.x,
		angular_velocity_local.y,
		angular_velocity_local.z,
		goal_area.get_relative_speed(),
	]

	for throttle_value in ship.get_thruster_controller().get_current_throttles():
		observation.append(throttle_value)

	observation.append(1.0 if goal_area.is_ship_inside() else 0.0)
	observation.append(1.0 if goal_area.is_goal_completed() else 0.0)
	return observation


func _get_goal_distance() -> float:
	return ship.global_position.distance_to(goal_area.global_position)


func _is_out_of_bounds() -> bool:
	if training_boundary_radius <= 0.0:
		return false

	return ship.global_position.distance_to(spawn_origin) > training_boundary_radius


func _coerce_float_array(raw_values: Variant) -> Array[float]:
	var action_values: Array[float] = []
	if typeof(raw_values) != TYPE_ARRAY:
		return action_values

	for raw_value in raw_values:
		action_values.append(float(raw_value))
	return action_values


func _send_error(code: String, message: String) -> void:
	_send_response({
		"ok": false,
		"error": code,
		"message": message,
	})


func _send_response(payload: Dictionary) -> void:
	if client == null:
		return

	var encoded_payload := (JSON.stringify(payload) + "\n").to_utf8_buffer()
	client.put_data(encoded_payload)