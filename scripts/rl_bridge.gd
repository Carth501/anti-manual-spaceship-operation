class_name RLBridge
extends Node

# RLBridge turns the current scene into a small synchronous RL environment.
# A Python process connects over TCP, sends line-delimited JSON commands,
# and receives observation/reward/done payloads back from Godot.

@export var ship: MiracleShip
@export var goal_area: GoalArea
@export var auto_start_server := true
@export var listen_host := "127.0.0.1"
@export_range(1024, 65535, 1) var listen_port := 8765
@export_range(1, 60, 1) var default_action_frames := 8
@export_range(1, 100000, 1) var episode_frame_limit := 2400
@export_range(0.0, 100000.0, 0.1, "or_greater") var training_boundary_radius := 1500.0
@export_range(0.0, 100.0, 0.001, "or_greater") var progress_reward_scale := 0.5
@export_range(0.0, 100000.0, 0.1, "or_greater") var approach_bonus_radius := 120.0
@export_range(0.0, 1000.0, 0.001, "or_greater") var approach_bonus_scale := 25.0
@export_range(0.0, 100.0, 0.001, "or_greater") var speed_penalty_scale := 0.02
@export_range(0.0, 100000.0, 0.1, "or_greater") var speed_penalty_distance := 60.0
@export_range(0.0, 100.0, 0.001, "or_greater") var thruster_penalty_scale := 0.001
@export_range(0.0, 10.0, 0.0001, "or_greater") var living_penalty_per_frame := 0.001
@export_range(0.0, 10000.0, 0.1, "or_greater") var goal_zone_entry_bonus := 20.0
@export_range(0.0, 10000.0, 0.1, "or_greater") var success_reward := 150.0
@export_range(0.0, 10000.0, 0.1, "or_greater") var out_of_bounds_penalty := 25.0
@export_range(0.0, 10000.0, 0.1, "or_greater") var timeout_penalty := 15.0
@export var emit_step_debug_logs := false
@export_range(1, 10000, 1) var step_debug_log_interval := 1

# Network state for the external trainer connection.
var server := TCPServer.new()
var client: StreamPeerTCP
var control_interface: ControlInterface
var receive_buffer := ""

# Episode bookkeeping. A single Python `step` holds one thruster action for
# `pending_action_frames` physics ticks before we compute reward and respond.
var episode_frames := 0
var episode_index := -1
var episode_reward_total := 0.0
var last_step_reward := 0.0
var last_terminal_reason := "idle"
var last_action_values: Array[float] = []
var previous_goal_distance := 0.0
var previous_inside_goal := false
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
	# Poll networking in the regular process loop so socket I/O stays responsive
	# even when physics is paused waiting for the next external command.
	_poll_server()
	if pending_step_frames == 0:
		_poll_client_messages()


func _physics_process(_delta: float) -> void:
	# Physics stepping is what makes this feel like a Gym environment: the last
	# action stays latched for N physics frames, then we emit one response.
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
		last_terminal_reason = "connected"


func _poll_client_messages() -> void:
	if client == null:
		return

	client.poll()
	if client.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		_release_rl_control()
		client = null
		receive_buffer = ""
		last_terminal_reason = "disconnected"
		return

	var available_bytes := client.get_available_bytes()
	if available_bytes <= 0:
		return

	receive_buffer += client.get_utf8_string(available_bytes)
	# Messages are newline-delimited JSON objects. We accumulate bytes until we
	# can carve out one full command at a time.
	var separator_index := receive_buffer.find("\n")
	while separator_index >= 0:
		var message := receive_buffer.substr(0, separator_index).strip_edges()
		receive_buffer = receive_buffer.substr(separator_index + 1)
		if not message.is_empty():
			_handle_message(message)
		separator_index = receive_buffer.find("\n")


func _handle_message(message: String) -> void:
	# Supported commands mirror a minimal RL API:
	# - hello: discover schema and action count
	# - reset: reset the episode and fetch the initial observation
	# - step: apply one action for a fixed number of physics frames
	# - close: disconnect the external trainer cleanly
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
			last_terminal_reason = "closed"
			client.disconnect_from_host()
			client = null
			receive_buffer = ""
		_:
			_send_error("unknown_command", "Unknown command: %s" % command)


func _begin_step(command: Dictionary) -> void:
	if pending_step_frames > 0:
		_send_error("busy", "A step is already in progress")
		return

	# The Python side sends one throttle value per thruster. We hand those values
	# directly to the ship and defer the response until physics has advanced.
	var action_values := _coerce_float_array(command.get("action", []))
	last_action_values = action_values.duplicate()
	control_interface.set_rl_control_enabled(true)
	control_interface.set_rl_thruster_inputs(action_values)
	pending_action_frames = max(int(command.get("frames", default_action_frames)), 1)
	pending_step_frames = pending_action_frames
	last_terminal_reason = "stepping"


func _complete_step() -> void:
	# Reward is dense shaping plus terminal bonuses/penalties. This is the point
	# where one logical env.step() finishes and we answer the Python client.
	var reward_terms := _compute_reward_terms()
	var total_reward: float = float(reward_terms.get("dense_total", 0.0))
	var done := false
	var terminal_reason := "in_progress"
	var current_inside_goal := goal_area.is_ship_inside()
	var goal_entry_bonus := 0.0
	var applied_success_bonus := 0.0
	var applied_out_of_bounds_penalty := 0.0
	var applied_timeout_penalty := 0.0

	if current_inside_goal and not previous_inside_goal:
		goal_entry_bonus = goal_zone_entry_bonus
		total_reward += goal_entry_bonus

	if goal_area.is_goal_completed():
		applied_success_bonus = success_reward
		total_reward += applied_success_bonus
		done = true
		terminal_reason = "goal_reached"
	elif _is_out_of_bounds():
		applied_out_of_bounds_penalty = out_of_bounds_penalty
		total_reward -= applied_out_of_bounds_penalty
		done = true
		terminal_reason = "out_of_bounds"
	elif episode_frames >= episode_frame_limit:
		applied_timeout_penalty = timeout_penalty
		total_reward -= applied_timeout_penalty
		done = true
		terminal_reason = "timeout"

	reward_terms["goal_entry_bonus"] = goal_entry_bonus
	reward_terms["success_bonus"] = applied_success_bonus
	reward_terms["out_of_bounds_penalty"] = applied_out_of_bounds_penalty
	reward_terms["timeout_penalty"] = applied_timeout_penalty
	reward_terms["total"] = total_reward

	last_step_reward = total_reward
	episode_reward_total += total_reward
	last_terminal_reason = terminal_reason
	var debug_snapshot := _build_step_debug_snapshot(reward_terms, total_reward, terminal_reason)
	if emit_step_debug_logs and episode_frames % max(step_debug_log_interval, 1) == 0:
		_print_step_debug_snapshot(debug_snapshot)
	previous_goal_distance = _get_goal_distance()
	previous_inside_goal = current_inside_goal
	_send_response(_build_step_response(total_reward, done, terminal_reason, reward_terms, debug_snapshot))


func _reset_episode_state() -> void:
	# Reset makes the scene deterministic again: restore the ship pose, zero out
	# thrusters and velocities, clear goal state, and restart episode counters.
	episode_index += 1
	episode_reward_total = 0.0
	last_step_reward = 0.0
	last_terminal_reason = "reset"
	last_action_values.clear()
	control_interface.set_rl_control_enabled(true)
	control_interface.set_rl_thruster_inputs([])
	ship.reset_state()
	goal_area.reset_goal()
	goal_area.refresh_goal_state()
	episode_frames = 0
	previous_goal_distance = _get_goal_distance()
	previous_inside_goal = goal_area.is_ship_inside()
	pending_step_frames = 0
	pending_action_frames = 0


func _release_rl_control() -> void:
	if control_interface == null:
		return

	control_interface.set_rl_control_enabled(false)


func _build_hello_response() -> Dictionary:
	# `hello` lets the Python wrapper discover how many thrusters/actions exist
	# in the current scene before it constructs Gym spaces.
	var current_scene: Node = get_tree().current_scene
	var scene_path := ""
	if current_scene != null:
		scene_path = current_scene.scene_file_path

	return {
		"ok": true,
		"version": 1,
		"thruster_count": ship.get_thruster_controller().get_thruster_count(),
		"default_action_frames": default_action_frames,
		"episode_frame_limit": episode_frame_limit,
		"physics_ticks_per_second": Engine.physics_ticks_per_second,
		"scene_path": scene_path,
		"spawn_origin": _vector3_to_array(spawn_origin),
		"observation": _get_observation(),
		"observation_schema": _build_observation_schema(),
		"reward_config": _build_reward_config(),
		"goal_config": _build_goal_config(),
		"ship_config": _build_ship_config(),
		"thruster_config": _build_thruster_config(),
	}


func _build_observation_schema() -> Dictionary:
	var fields: Array[String] = [
		"goal_offset_local_x",
		"goal_offset_local_y",
		"goal_offset_local_z",
		"linear_velocity_local_x",
		"linear_velocity_local_y",
		"linear_velocity_local_z",
		"angular_velocity_local_x",
		"angular_velocity_local_y",
		"angular_velocity_local_z",
		"relative_speed",
	]
	var thruster_count := ship.get_thruster_controller().get_thruster_count()
	for index in range(thruster_count):
		fields.append("thruster_throttle_%02d" % index)
	fields.append("is_inside_goal")
	fields.append("is_goal_completed")
	return {
		"fields": fields,
		"fixed_field_count": 10,
		"thruster_field_count": thruster_count,
		"flag_field_count": 2,
	}


func _build_reward_config() -> Dictionary:
	return {
		"training_boundary_radius": training_boundary_radius,
		"progress_reward_scale": progress_reward_scale,
		"approach_bonus_radius": approach_bonus_radius,
		"approach_bonus_scale": approach_bonus_scale,
		"speed_penalty_scale": speed_penalty_scale,
		"speed_penalty_distance": speed_penalty_distance,
		"thruster_penalty_scale": thruster_penalty_scale,
		"living_penalty_per_frame": living_penalty_per_frame,
		"goal_zone_entry_bonus": goal_zone_entry_bonus,
		"success_reward": success_reward,
		"out_of_bounds_penalty": out_of_bounds_penalty,
		"timeout_penalty": timeout_penalty,
	}


func _build_goal_config() -> Dictionary:
	return {
		"goal_position": _vector3_to_array(goal_area.global_position),
		"goal_velocity": _vector3_to_array(goal_area.goal_velocity),
		"speed_threshold_mps": goal_area.speed_threshold_mps,
	}


func _build_ship_config() -> Dictionary:
	return {
		"mass": ship.mass,
		"linear_damp": ship.linear_damp,
		"angular_damp": ship.angular_damp,
		"gravity_scale": ship.gravity_scale,
		"spawn_position": _vector3_to_array(ship.global_position),
		"spawn_basis": _basis_to_rows(ship.global_transform.basis),
	}


func _build_thruster_config() -> Dictionary:
	var controller: ThrusterController = ship.get_thruster_controller()
	var thrusters: Array[Dictionary] = []
	if controller != null:
		for index in range(controller.thrusters.size()):
			var thruster: ThrusterPoint = controller.thrusters[index]
			if thruster == null:
				continue
			thrusters.append({
				"index": index,
				"name": thruster.name,
				"enabled": thruster.enabled,
				"position_local": _vector3_to_array(controller.to_local(thruster.global_position)),
				"thrust_direction_local": _vector3_to_array(thruster.thrust_direction.normalized()),
				"linear_response": thruster.linear_response,
				"angular_response": thruster.angular_response,
				"max_force": thruster.max_force,
			})
	return {
		"center_of_mass_local": _vector3_to_array(controller.center_of_mass_local),
		"direct_throttle_slew_rate": controller.direct_throttle_slew_rate,
		"thrusters": thrusters,
	}


func _vector3_to_array(value: Vector3) -> Array[float]:
	var result: Array[float] = [value.x, value.y, value.z]
	return result


func _basis_to_rows(value: Basis) -> Array[Array]:
	var rows: Array[Array] = []
	rows.append(_vector3_to_array(value.x))
	rows.append(_vector3_to_array(value.y))
	rows.append(_vector3_to_array(value.z))
	return rows


func _build_step_response(
		reward: float,
		done: bool,
		terminal_reason: String,
		reward_terms: Dictionary = {},
		debug_snapshot: Dictionary = {}
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
			"debug": debug_snapshot,
		},
	}


func _compute_reward_terms() -> Dictionary:
	# These terms are intentionally simple for the first curriculum:
	# move toward the goal, keep relative speed low, and avoid wasting thrust.
	var current_goal_distance := _get_goal_distance()
	var goal_distance_delta := previous_goal_distance - current_goal_distance
	var progress_reward := goal_distance_delta * progress_reward_scale
	var previous_goal_potential := _compute_goal_potential(previous_goal_distance)
	var current_goal_potential := _compute_goal_potential(current_goal_distance)
	var approach_bonus := (current_goal_potential - previous_goal_potential) * approach_bonus_scale
	var speed_penalty_weight := _get_speed_penalty_weight(current_goal_distance)
	var speed_penalty := goal_area.get_relative_speed() * speed_penalty_scale * speed_penalty_weight
	var throttles := ship.get_thruster_controller().get_current_throttles()
	var throttle_sum := 0.0
	for throttle_value in throttles:
		throttle_sum += throttle_value
	var thruster_penalty := throttle_sum * thruster_penalty_scale
	var living_penalty := float(max(pending_action_frames, 1)) * living_penalty_per_frame
	var total_reward := progress_reward + approach_bonus - speed_penalty - thruster_penalty - living_penalty

	return {
		"goal_distance_delta": goal_distance_delta,
		"previous_goal_potential": previous_goal_potential,
		"current_goal_potential": current_goal_potential,
		"progress": progress_reward,
		"approach_bonus": approach_bonus,
		"speed_penalty": speed_penalty,
		"speed_penalty_weight": speed_penalty_weight,
		"thruster_penalty": thruster_penalty,
		"living_penalty": living_penalty,
		"dense_total": total_reward,
		"total": total_reward,
	}


func _compute_goal_potential(goal_distance: float) -> float:
	if approach_bonus_radius <= 0.0:
		return 0.0

	return exp(-goal_distance / approach_bonus_radius)


func _get_speed_penalty_weight(goal_distance: float) -> float:
	if goal_area.is_ship_inside():
		return 1.0

	if speed_penalty_distance <= 0.0:
		return 1.0

	return clamp(1.0 - (goal_distance / speed_penalty_distance), 0.0, 1.0)


func _get_observation() -> Array[float]:
	# The observation is expressed mostly in ship-local coordinates so the policy
	# learns relative control effects instead of memorizing world orientation.
	# Layout: goal offset, linear velocity, angular velocity, per-thruster
	# throttles, inside-goal flag, goal-complete flag.
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


func get_training_hud_state() -> Dictionary:
	return {
		"connected": is_trainer_connected(),
		"phase": _get_training_phase(),
		"episode_index": max(episode_index, 0),
		"episode_frames": episode_frames,
		"episode_reward_total": episode_reward_total,
		"last_step_reward": last_step_reward,
		"last_terminal_reason": last_terminal_reason,
	}


func _build_step_debug_snapshot(reward_terms: Dictionary, total_reward: float, terminal_reason: String) -> Dictionary:
	var goal_offset_local := ship.get_local_vector(goal_area.global_position - ship.global_position)
	var linear_velocity_local := ship.get_local_vector(ship.get_linear_velocity_readout())
	var angular_velocity_local := ship.get_local_vector(ship.get_angular_velocity_readout())
	var applied_throttles := ship.get_thruster_controller().get_current_throttles()
	var throttle_sum := 0.0
	var throttle_max := 0.0
	for throttle_value in applied_throttles:
		throttle_sum += throttle_value
		throttle_max = max(throttle_max, throttle_value)

	return {
		"episode_index": max(episode_index, 0),
		"episode_frames": episode_frames,
		"action_request": last_action_values.duplicate(),
		"applied_throttles": applied_throttles,
		"throttle_sum": throttle_sum,
		"throttle_max": throttle_max,
		"goal_offset_local": [goal_offset_local.x, goal_offset_local.y, goal_offset_local.z],
		"linear_velocity_local": [linear_velocity_local.x, linear_velocity_local.y, linear_velocity_local.z],
		"angular_velocity_local": [angular_velocity_local.x, angular_velocity_local.y, angular_velocity_local.z],
		"goal_distance": _get_goal_distance(),
		"relative_speed": goal_area.get_relative_speed(),
		"is_inside_goal": goal_area.is_ship_inside(),
		"is_goal_completed": goal_area.is_goal_completed(),
		"reward_terms": reward_terms,
		"reward_total": total_reward,
		"terminal_reason": terminal_reason,
	}


func _print_step_debug_snapshot(debug_snapshot: Dictionary) -> void:
	print("[RLBridge] %s" % JSON.stringify(debug_snapshot))


func is_trainer_connected() -> bool:
	return client != null and client.get_status() == StreamPeerTCP.STATUS_CONNECTED


func _get_training_phase() -> String:
	if pending_step_frames > 0:
		return "stepping"

	if is_trainer_connected():
		return "waiting_command"

	return "idle"


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