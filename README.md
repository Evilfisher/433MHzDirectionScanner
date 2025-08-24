Never Plug in the USB Connection if 5v are supplied externally, Use only one powersource at a time.

Using an ESP32, a 433 MHz receiver, and an SG90 servo, I built a small direction-scanning rig. 
The ESP32 sweeps horizontally, logs RF activity, sends it via Wi-Fi to a Flask server, and a Python dashboard shows a radar view, heatmap, and pulse spectrum in real time. 
A simple classifier estimates whether signals come from things like weather sensors or fixed-code remotes. I’m genuinely proud of how cleanly it all works.
What it does:
Servo (0–180°) sweeps the antenna; ESP32 reads the 433 MHz RX line and timestamps edges in microseconds.
Data is posted as CSV to http://<PC>:8000/ingest.

A live Python dashboard (Matplotlib) renders:
Direction radar (polar plot)
Heatmap (time × angle)
Pulse spectrum (histogram)

A heuristic “device classifier” uses unit pulse width, packet length, repeats per press, and periodicity to label signals (e.g., weather/home sensor, smart plug/doorbell fixed-code, rolling-code remote), with confidence and rationale.
A simple phone viewer serves the latest PNG. Session mode with /reset clears CSV/PNGs and starts fresh when moving the setup.

Hardware
ESP32 dev board
433 MHz RX module (WayinTop) with spring antenna
SG90 servo (5 V)
5 V supply, common ground, decoupling recommended
Firmware (Arduino / ESP32)
Libraries: WiFi.h, HTTPClient.h, ESP32Servo.h
Example pins: Servo = GPIO14, RF data = GPIO27
Small sweep step; on signal, short hold; CSV POST as Timestamp,Angle,Signal or edges as t_us,level,angle.
PC side (Python)
Flask server: /ingest, /latest.png, /reset (session cleanup)
Analyzer/Dashboard: pandas, numpy, matplotlib
Dark “console” style with optional world-map background

Built iteratively with ChatGPT for code scaffolding, heuristics, and UI tweaks. Feedback and questions welcome.
Note: this is passive reception and analysis. Only analyze your own devices and follow local regulations.

First step: Upload Sketch and wire the ESP32 according to the selected Pins in the Sketch
1.Start Ingest Server on your computer in a Command prompt or VS Code Studio
2.Start viewer Drawing in a Command prompt with: python .\analyze_signals.py --classify --gap 8000 --classify-window 45 --save-every 1 --bg world_hud.png --bg-alpha 0.18
3.Power on the ESP32 and let it scan, either power via 5v supply ord usb cable, data is transmitted wireless

Browse to your Computer/Server IP and watch the dashboard, alternatively you can also watch it on your mobile, just browse to the correct ip
