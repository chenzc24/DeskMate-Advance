# Network Endpoints

All changeable runtime IP addresses, URLs and service ports are defined in:

`configs/runtime/network_endpoints.json`

`bind_host` controls which Windows interfaces accept connections.
`advertised_host` is the Windows address entered in the phone browser. The
phone URL is derived as
`http://<advertised_host>:<port>/`. `camera_streams.robot_camera.url` is the
small-car/Raspberry Pi MJPEG endpoint.

The `robot_camera`, `robot_camera_audiorelay`, and `robot_hardware` runtime
profiles contain only `"stream_endpoint": "robot_camera"` and resolve the
actual URL from this file. Changing Wi-Fi therefore requires editing only this
one configuration.

The standard runtime reads this file automatically:

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile robot_camera_audiorelay `
  --mode live `
  --button seat_a `
  --consent-confirmed `
  --announcer windows `
  --web-console `
  --headless
```

Use `--network-config <path>` to select another complete endpoint set. The
`--stream-url`, `--web-host`, and `--web-port` options remain temporary
single-run overrides and do not modify the shared file.

To inspect the currently configured values without copying them into another
file:

```powershell
$network = Get-Content configs/runtime/network_endpoints.json | ConvertFrom-Json
"Phone: http://$($network.mobile_web_console.advertised_host):$($network.mobile_web_console.port)/"
"Camera: $($network.camera_streams.robot_camera.url)"
```

The configuration is validated against
`configs/contracts/network_endpoints.schema.json`. Camera URLs must be absolute
HTTP(S) URLs without embedded credentials, query parameters or fragments. The
advertised phone host may not be a wildcard address.
