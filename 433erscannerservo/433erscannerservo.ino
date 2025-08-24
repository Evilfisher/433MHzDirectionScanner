/***** ESP32: 433MHz Edge-Logger + Servo Scanner + CSV Upload *****/
#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ESP32Servo.h>

// ===== WLAN & Server =====
const char* ssid      = "Your-SSID";
const char* password  = "Your-Wifi-PW";
const char* serverUrl = "http://192.168.1.13:8000/ingest"; // Flask-Server IP

// ===== Pins =====
#define RX_PIN     27   // 433 MHz Empfänger DATA
#define SERVO_PIN  14   // SG90 Signal

// ===== Scan-Parameter =====
int   STEP_DEG         = 10;     // Schrittweite Servo
int   DWELL_MS         = 400;    // Wartezeit nach Servobewegung
int   PROBE_MS         = 2000;   // Messzeit pro Winkel (Signalcheck)
int   MIN_PULSES_TRIG  = 5;      // ab so vielen Flanken: „Signal vorhanden“
int   ANALYZE_MS       = 10000;  // Analyse-/Logdauer (ms)

// ===== Edge-Puffer (ISR) =====
static const size_t BUF_CAP = 4096;
volatile uint32_t   t_buf[BUF_CAP];
volatile uint8_t    lvl_buf[BUF_CAP];
volatile size_t     w_head = 0;
portMUX_TYPE isrMux = portMUX_INITIALIZER_UNLOCKED;

Servo scanner;

void IRAM_ATTR rxISR() {
  uint32_t now = micros();
  uint8_t level = (uint8_t)digitalRead(RX_PIN);
  portENTER_CRITICAL_ISR(&isrMux);
  if (w_head < BUF_CAP) {
    t_buf[w_head]   = now;
    lvl_buf[w_head] = level;
    w_head++;
  } else {
    // overflow -> drop
  }
  portEXIT_CRITICAL_ISR(&isrMux);
}

void clearBuffer() {
  portENTER_CRITICAL(&isrMux);
  w_head = 0;
  portEXIT_CRITICAL(&isrMux);
}

bool wifiEnsureConnected() {
  if (WiFi.status() == WL_CONNECTED) return true;
  WiFi.disconnect(true);
  WiFi.begin(ssid, password);
  Serial.print("WLAN verbinden");
  for (int i = 0; i < 60 && WiFi.status() != WL_CONNECTED; i++) {
    delay(250); Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("✔ IP: "); Serial.println(WiFi.localIP());
    return true;
  }
  Serial.println("✖ WLAN fehlgeschlagen");
  return false;
}

bool postCSV(const String& csvChunk) {
  if (csvChunk.length() == 0) return true;
  if (!wifiEnsureConnected()) return false;
  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "text/csv");
  int code = http.POST((uint8_t*)csvChunk.c_str(), csvChunk.length());
  http.end();
  Serial.printf("→ CSV %u Bytes, HTTP %d\n", csvChunk.length(), code);
  return (code >= 200 && code < 300);
}

size_t snapshotToCSV(String& out, uint16_t angle, uint32_t t0) {
  size_t count = 0;
  portENTER_CRITICAL(&isrMux);
  size_t n = w_head;
  if (n > BUF_CAP) n = BUF_CAP;
  for (size_t i = 0; i < n; i++) {
    uint32_t dt = t_buf[i] - t0;  // relative Zeit
    out += String(dt); out += ",";
    out += String((int)lvl_buf[i]); out += ",";
    out += String((int)angle); out += "\n";
    count++;
  }
  w_head = 0;
  portEXIT_CRITICAL(&isrMux);
  return count;
}

bool detectActivityAtAngle(int angle) {
  scanner.write(angle);
  delay(DWELL_MS);
  clearBuffer();
  delay(PROBE_MS);  // Probezeit
  size_t n;
  portENTER_CRITICAL(&isrMux);
  n = w_head;
  portEXIT_CRITICAL(&isrMux);
  Serial.printf("Winkel %3d° → %u Flanken in %d ms\n", angle, (unsigned)n, PROBE_MS);
  return (n >= (size_t)MIN_PULSES_TRIG);
}

void analyzeAndUpload(int angle) {
  Serial.printf("📡 Signal @ %d° → sammle %d s Rohdaten …\n", angle, ANALYZE_MS/1000);
  uint32_t t0 = micros();
  clearBuffer();

  uint32_t startMs = millis();
  while (millis() - startMs < (uint32_t)ANALYZE_MS) {
    delay(250); // alle 250 ms flushen
    String csv;
    // Header nur beim ersten Chunk (optional)
    // csv += "t_us,level,angle\n";
    size_t n = snapshotToCSV(csv, angle, t0);
    if (n) postCSV(csv);
  }
  // Rest flushen
  String tail;
  size_t n = snapshotToCSV(tail, angle, t0);
  if (n) postCSV(tail);
  Serial.println("✅ Analyse abgeschlossen – scanne weiter.");
}

void setup() {
  Serial.begin(115200);
  pinMode(RX_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(RX_PIN), rxISR, CHANGE);

  // Servo (SG90 50Hz, 500–2400µs)
  ESP32PWM::allocateTimer(0);
  scanner.setPeriodHertz(50);
  scanner.attach(SERVO_PIN, 500, 2400);

  // 10 s Mittelstellung für Ausrichtung
  scanner.write(90);
  Serial.println("⏳ Servo Mittelstellung (90°) für 10 s …");
  delay(10000);

  wifiEnsureConnected();
}

void loop() {
  // 0..180
  for (int angle = 0; angle <= 180; angle += STEP_DEG) {
    if (detectActivityAtAngle(angle)) analyzeAndUpload(angle);
  }
  // 180..0
  for (int angle = 180; angle >= 0; angle -= STEP_DEG) {
    if (detectActivityAtAngle(angle)) analyzeAndUpload(angle);
  }
}
