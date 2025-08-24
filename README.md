Upload Sketch and wire the ESP32 according to the selected Pins in the Sketch
1.Start Ingest Server on your computer in a Command prompt or VS Code Studio
2.Start viewer Drawing in a Command prompt with: python .\analyze_signals.py --classify --gap 8000 --classify-window 45 --save-every 1 --bg world_hud.png --bg-alpha 0.18
3.Plug in ESP32 and let it scan
Browse to your Computer/Server IP and watch the dashboard, alternatively you can also watch it on your mobile, just browse to the correct ip
