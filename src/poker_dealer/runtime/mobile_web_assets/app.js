(() => {
  const roles = ["button", "small_blind", "big_blind", "under_the_gun"];
  const roleLabels = {
    button: "Button",
    small_blind: "Small blind",
    big_blind: "Big blind",
    under_the_gun: "Under the gun",
  };
  const registrationPhases = {
    starting: ["1", "Starting", "Waiting for registration state."],
    ready_for_face: ["1", "Align one face", "Keep one player in view, then confirm capture."],
    capturing_face: ["2", "Capturing face", "Keep the current player centered and steady."],
    ready_to_start: ["3", "Roster ready", "All four faces are enrolled. Start the game."],
    started: ["4", "Registration complete", "The face roster is frozen for this session."],
  };
  const phaseDescriptions = {
    dealing_hole: "Dealing two face-down hole cards to every player.",
    awaiting_action: "Waiting for the focused player’s voice or hand action.",
    dealing_board: "Recognizing the current face-up community card.",
    showdown: "Revealing live players and determining the winner.",
    settled: "The pot has been awarded. Return all cards.",
    paused_recovery: "The hand is paused. Resolve the displayed issue.",
    voided: "The hand was voided.",
  };
  const suitSymbols = {
    clubs: "♣", diamonds: "♦", hearts: "♥", spades: "♠",
    club: "♣", diamond: "♦", heart: "♥", spade: "♠",
  };
  const $ = (id) => document.getElementById(id);
  let socket;
  let viewVersion = 0;
  let controller = false;
  let allowed = new Set();
  let pendingIntent = null;
  let latestFaceBoxes = [];
  let latestFaceStatus = "";
  let latestActionMarker = null;
  let commandPending = false;
  let phoneVoiceEnabled = true;

  function setLink(text, mode = "") {
    $("linkStatus").className = `status ${mode}`.trim();
    $("linkStatus").lastChild.textContent = ` ${text}`;
  }

  function connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws`);
    socket.addEventListener("open", () => setLink("Connected"));
    socket.addEventListener("close", () => {
      controller = false;
      setLink("Reconnecting", "warning");
      updateButtons();
      setTimeout(connect, 1200);
    });
    socket.addEventListener("error", () => setLink("Offline", "offline"));
    socket.addEventListener("message", ({ data }) => {
      const message = JSON.parse(data);
      if (message.type === "hello" || message.type === "controller_changed") {
        controller = Boolean(message.controller);
        $("feedback").textContent = controller
          ? "This phone has control."
          : "Viewer mode — another device has control.";
        updateButtons();
      } else if (message.type === "state") {
        controller = Boolean(message.controller);
        viewVersion = message.view_version;
        allowed = new Set(message.allowed_intents || []);
        renderState(message);
      } else if (message.type === "command_ack") {
        const okay = message.status === "queued" || message.status === "accepted";
        commandPending = false;
        $("feedback").textContent = okay
          ? "Command sent to the Windows runtime."
          : `Not sent: ${humanize(message.reason)}`;
        updateButtons();
      } else if (message.type === "control_result") {
        commandPending = false;
        $("feedback").textContent = message.accepted
          ? "Command accepted."
          : `Runtime rejected: ${humanize(message.reason)}`;
        updateButtons();
      } else if (message.type === "prompt") {
        $("feedback").textContent = message.text;
        speakPrompt(message.text);
      }
    });
  }

  function renderState(message) {
    const state = message.state || {};
    const view = state.view || "registration";
    document.querySelectorAll(".registration-only").forEach((node) => {
      node.hidden = view !== "registration";
    });
    $("gamePanel").hidden = view === "registration";
    if (view === "registration") {
      renderRegistration(state);
    } else if (view === "hand") {
      renderHand(state);
    } else {
      renderSessionBoundary(state);
    }
    $("cameraStatus").classList.toggle("offline", !message.video_ready);
    $("alert").hidden = !state.alert_title && !state.paused_reason;
    $("alertTitle").textContent = state.alert_title || (state.paused_reason ? "Recovery required" : "");
    $("alertDetail").textContent = state.alert_detail || humanize(state.paused_reason || "");
    const simulated = isSimulatedActingSeat(state);
    const runtimeFaceVisible = (
      view === "hand"
      && state.phase === "awaiting_action"
      && !simulated
    );
    const faceStatus = view === "registration"
      ? (message.face_status || (message.video_ready ? "Waiting for one face" : "Waiting for camera"))
      : simulated
      ? `${seatShort(state.acting_seat)} SIMULATED · AUTO FOLD`
      : runtimeFaceVisible
      ? (message.face_status || runtimeFaceFallback(state))
      : `${humanize(state.phase || "live")} · ${humanize(state.current_target || state.acting_seat || "")}`;
    $("faceStatus").textContent = faceStatus;
    renderFaceBoxes(
      view === "registration" || runtimeFaceVisible ? (message.face_boxes || []) : [],
      faceStatus,
      runtimeFaceVisible ? message.action_marker : null,
    );
    document.querySelector("[data-intent='confirm']").hidden = view === "hand";
    if (message.runtime_feedback) $("feedback").textContent = message.runtime_feedback;
    updateButtons();
  }

  function renderRegistration(state) {
    const phase = registrationPhases[state.phase] || registrationPhases.starting;
    $("consoleTitle").textContent = "Player registration";
    $("stepNumber").textContent = state.voice_active ? "3" : phase[0];
    $("stepTitle").textContent = state.voice_active ? "Recording voice" : phase[1];
    $("stepDetail").textContent = state.voice_active ? voiceInstruction(state) : phase[2];
    $("role").textContent = roleLabels[state.role] || humanize(state.role);
    $("seat").textContent = `Seat ${String(state.seat || "—").replace("seat_", "").toUpperCase()}`;
    setProgress("face", state.face_samples, state.face_target);
    setProgress("voice", state.voice_samples, state.voice_target);
    const voiceEnrollmentEnabled = Number(state.voice_target || 0) > 0;
    document.querySelector(".voice-progress").hidden = !voiceEnrollmentEnabled;
    document.querySelector(".mic-progress").hidden = !voiceEnrollmentEnabled;
    $("micStatus").classList.toggle("offline", !state.microphone_live);
    $("micStatus").lastChild.textContent = state.microphone_live ? " Mic live" : " Mic offline";
    renderMicLevel(state.microphone_level || 0, state.microphone_live);
    renderRoster(state.completed_roles || [], state.simulated_roles || []);
    $("confirmLabel").textContent = "Capture";
    $("startLabel").textContent = "Start";
    $("clearLabel").textContent = "Clear";
  }

  function renderHand(state) {
    $("consoleTitle").textContent = `Hand ${state.hand_id || ""}`.trim();
    $("stepNumber").textContent = streetIndex(state.street, state.phase);
    $("stepTitle").textContent = humanize(state.phase || "hand");
    $("stepDetail").textContent = handDetail(state);
    $("gameStreet").textContent = humanize(state.street || "pre-deal");
    $("gamePot").textContent = String(state.pot_units ?? 0);
    $("gameActing").textContent = seatLabel(state.acting_seat, state.players_by_seat);
    $("gameTarget").textContent = humanize(state.current_target || "—");
    renderCards(state.board || []);
    renderPlayerLedger(state.players || {}, state.players_by_seat || {}, state.acting_seat);
    renderLegalActions(state.legal_actions || []);
    $("legalActionsBlock").hidden = state.phase !== "awaiting_action";
    $("confirmLabel").textContent = "Confirm action";
    $("startLabel").textContent = "Retry";
    $("clearLabel").textContent = "Void hand";
  }

  function renderSessionBoundary(state) {
    $("consoleTitle").textContent = state.ended ? "Session complete" : "Session control";
    $("stepNumber").textContent = state.phase === "recovery" ? "!" : "↻";
    $("stepTitle").textContent = humanize(state.phase || "session");
    $("stepDetail").textContent = sessionDetail(state);
    $("gameStreet").textContent = "Between hands";
    $("gamePot").textContent = "0";
    $("gameActing").textContent = "—";
    $("gameTarget").textContent = state.button ? `Next ${seatLabel(state.button)}` : "—";
    renderCards([]);
    const players = {};
    Object.entries(state.stacks || {}).forEach(([seat, stack]) => {
      players[seat] = { stack_units: stack, street_commit_units: 0, hand_commit_units: 0 };
    });
    renderPlayerLedger(players, {}, null);
    renderLegalActions([]);
    $("legalActionsBlock").hidden = true;
    $("confirmLabel").textContent = state.phase === "table_clearance" ? "Table clear" : "Confirm";
    $("startLabel").textContent = state.phase === "recovery" ? "Retry hand" : "Next hand";
    $("clearLabel").textContent = state.phase === "recovery" ? "Void hand" : "End session";
  }

  function handDetail(state) {
    if (state.part_a_phase === "verifying_identity") {
      return `Verifying Seat ${seatShort(state.acting_seat)} automatically. Keep one face in view.`;
    }
    if (state.part_a_phase === "waiting_player_action") {
      return `Seat ${seatShort(state.acting_seat)} verified. Say one legal English action clearly.`;
    }
    const lane = state.part_a_phase || state.part_b_phase;
    const detail = phaseDescriptions[state.phase] || "Synchronizing game state.";
    return lane ? `${detail} Runtime gate: ${humanize(lane)}.` : detail;
  }

  function sessionDetail(state) {
    if (state.phase === "recovery") {
      const slot = state.selected_slot ? ` Selected slot: ${humanize(state.selected_slot)}.` : "";
      return `Paused: ${humanize(state.paused_reason || "unknown")}.${slot} Retry only after state parity is checked.`;
    }
    if (state.phase === "table_clearance") return "Return every card, then confirm that the table is clear.";
    if (state.phase === "ready_next_hand") {
      const seat = state.selected_seat ? ` Selected rebuy seat: ${seatLabel(state.selected_seat)}.` : "";
      return `Start the next hand, adjust a selected low stack, or end the session.${seat}`;
    }
    return "The session has ended.";
  }

  function streetIndex(street, phase) {
    if (phase === "dealing_hole") return "0";
    return { preflop: "1", flop: "2", turn: "3", river: "4" }[street] || "5";
  }

  function seatLabel(seat, players = {}) {
    if (!seat) return "—";
    const short = String(seat).replace("seat_", "").toUpperCase();
    const player = players?.[seat];
    return player ? `Seat ${short} · ${player}` : `Seat ${short}`;
  }

  function seatShort(seat) {
    return String(seat || "?").replace("seat_", "").toUpperCase();
  }

  function isSimulatedActingSeat(state) {
    const player = state.players_by_seat?.[state.acting_seat];
    return String(player || "").startsWith("development-simulator-");
  }

  function runtimeFaceFallback(state) {
    const seat = seatShort(state.acting_seat);
    if (state.part_a_phase === "waiting_player_action") {
      return `${seat} VERIFIED · LISTENING`;
    }
    if (state.part_a_phase === "verifying_identity") return `VERIFYING ${seat}`;
    if (state.part_a_phase === "waiting_visual_settle") return `POSITIONING ${seat}`;
    return `WAITING FOR ${seat}`;
  }

  function cardLabel(card) {
    const rank = String(card?.rank || "?").replace("10", "10").toUpperCase();
    const suit = suitSymbols[String(card?.suit || "").toLowerCase()] || "?";
    return `${rank}${suit}`;
  }

  function renderCards(cards) {
    const values = cards.length ? cards.map(cardLabel) : ["—", "—", "—", "—", "—"];
    $("boardCards").replaceChildren(...values.map((value) => {
      const node = document.createElement("span");
      node.textContent = value;
      return node;
    }));
  }

  function renderPlayerLedger(players, ids, actingSeat) {
    const order = ["seat_a", "seat_b", "seat_c", "seat_d"];
    $("playerLedger").replaceChildren(...order.map((seat) => {
      const player = players[seat] || {};
      const node = document.createElement("div");
      node.className = "ledger-seat";
      node.classList.toggle("acting", seat === actingSeat);
      node.classList.toggle("folded", Boolean(player.folded));
      const name = document.createElement("strong");
      name.textContent = seatLabel(seat, ids);
      const balance = document.createElement("span");
      balance.textContent = `Stack ${player.stack_units ?? "—"} · Bet ${player.street_commit_units ?? 0} · Hand ${player.hand_commit_units ?? 0}`;
      const status = document.createElement("span");
      status.textContent = player.folded ? "Folded" : player.all_in ? "All-in" : seat === actingSeat ? "Acting" : "Waiting";
      node.append(name, balance, status);
      return node;
    }));
  }

  function renderLegalActions(actions) {
    const values = actions.length ? actions : ["None"];
    $("legalActions").replaceChildren(...values.map((value) => {
      const node = document.createElement("span");
      node.textContent = humanize(value);
      return node;
    }));
  }

  function setProgress(prefix, count = 0, target = 0) {
    $(prefix + "Count").textContent = `${count} / ${target}`;
    $(prefix + "Progress").style.width = `${target ? Math.min(100, count / target * 100) : 0}%`;
  }

  function renderMicLevel(level, live) {
    const decibels = level > 0 ? 20 * Math.log10(level) : -60;
    const percent = Math.max(0, Math.min(100, (decibels + 60) / 60 * 100));
    $("micLevel").style.width = `${percent}%`;
    $("micLevelText").textContent = live
      ? (level > 0.01 ? "Voice detected" : "Connected · silent")
      : "Disconnected";
  }

  function voiceInstruction(state) {
    if (state.prompt_playing) return "Prompt is playing. Wait until it finishes.";
    if (!state.microphone_live) return "AudioRelay microphone is disconnected.";
    const phrases = ["CHECK", "CALL", "RAISE"];
    const phrase = phrases[Math.min(state.voice_samples || 0, 2)];
    return `Say “${phrase}”, then pause.`;
  }

  function renderRoster(completed, simulated) {
    const done = new Set(completed);
    const simulatedRoles = new Set(simulated);
    $("roster").replaceChildren(...roles.map((role) => {
      const item = document.createElement("span");
      item.textContent = `${done.has(role) ? "✓" : "○"} ${roleLabels[role]}`;
      item.classList.toggle("done", done.has(role));
      if (simulatedRoles.has(role)) item.textContent += " · SIMULATED";
      item.classList.toggle("simulated", simulatedRoles.has(role));
      return item;
    }));
  }

  function renderFaceBoxes(boxes, status = "", actionMarker = null) {
    latestFaceBoxes = boxes;
    latestFaceStatus = status;
    latestActionMarker = actionMarker;
    const layer = $("faceLayer");
    const image = $("video");
    const stage = $("videoStage");
    const frameRatio = image.naturalWidth && image.naturalHeight
      ? image.naturalWidth / image.naturalHeight
      : stage.clientWidth / stage.clientHeight;
    const stageRatio = stage.clientWidth / stage.clientHeight;
    let displayWidth = stage.clientWidth;
    let displayHeight = stage.clientHeight;
    let offsetX = 0;
    let offsetY = 0;
    if (frameRatio > stageRatio) {
      displayHeight = displayWidth / frameRatio;
      offsetY = (stage.clientHeight - displayHeight) / 2;
    } else {
      displayWidth = displayHeight * frameRatio;
      offsetX = (stage.clientWidth - displayWidth) / 2;
    }
    const overlays = boxes.map((box) => {
      const node = document.createElement("div");
      node.className = "face-box";
      node.classList.toggle(
        "verified",
        status.includes("VERIFIED") || status.includes("HEARD"),
      );
      node.classList.toggle(
        "lost",
        status.includes("LOST") || status.includes("WRONG"),
      );
      node.style.left = `${offsetX + box.x * displayWidth}px`;
      node.style.top = `${offsetY + box.y * displayHeight}px`;
      node.style.width = `${box.width * displayWidth}px`;
      node.style.height = `${box.height * displayHeight}px`;
      return node;
    });
    if (actionMarker) {
      const marker = document.createElement("div");
      marker.className = "action-marker";
      marker.title = `${humanize(actionMarker.action)} detected`;
      marker.style.left = `${offsetX + actionMarker.x * displayWidth}px`;
      marker.style.top = `${offsetY + actionMarker.y * displayHeight}px`;
      overlays.push(marker);
    }
    layer.replaceChildren(...overlays);
  }

  function updateButtons() {
    document.querySelectorAll("[data-intent]").forEach((button) => {
      const intent = button.dataset.intent;
      button.disabled = !controller || socket?.readyState !== WebSocket.OPEN ||
        commandPending ||
        (!["quit", "repeat_prompt"].includes(intent) && !allowed.has(intent));
    });
  }

  function send(intent) {
    if (!controller || socket?.readyState !== WebSocket.OPEN) return;
    commandPending = true;
    updateButtons();
    try {
      socket.send(JSON.stringify({
        type: "command",
        command_id: createCommandId(),
        intent,
        expected_view_version: viewVersion,
      }));
    } catch (error) {
      commandPending = false;
      $("feedback").textContent = "Command could not be sent. Refresh the page.";
      updateButtons();
    }
  }

  document.querySelectorAll("[data-intent]").forEach((button) => {
    button.addEventListener("click", () => {
      const confirmation = button.dataset.confirm;
      if (!confirmation) return send(button.dataset.intent);
      pendingIntent = button.dataset.intent;
      $("confirmText").textContent = confirmation;
      $("confirmDialog").showModal();
    });
  });
  $("confirmDialog").addEventListener("close", () => {
    if ($("confirmDialog").returnValue === "confirm" && pendingIntent) send(pendingIntent);
    pendingIntent = null;
  });
  $("confirmButton").addEventListener("click", () => {
    $("confirmDialog").returnValue = "confirm";
  });
  $("video").addEventListener("load", () => (
    renderFaceBoxes(latestFaceBoxes, latestFaceStatus, latestActionMarker)
  ));
  window.addEventListener("resize", () => (
    renderFaceBoxes(latestFaceBoxes, latestFaceStatus, latestActionMarker)
  ));
  $("voiceToggle").addEventListener("click", () => {
    phoneVoiceEnabled = !phoneVoiceEnabled;
    $("voiceToggle").textContent = phoneVoiceEnabled ? "Phone voice on" : "Phone voice off";
    if (!phoneVoiceEnabled) window.speechSynthesis?.cancel();
  });

  function humanize(value) {
    const text = String(value || "").replaceAll("_", " ");
    return text ? text.charAt(0).toUpperCase() + text.slice(1) : "—";
  }

  function createCommandId() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    return `web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
  }

  function speakPrompt(text) {
    if (!phoneVoiceEnabled || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 0.95;
    utterance.volume = 1;
    window.speechSynthesis.speak(utterance);
  }

  connect();
})();
