#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <RTClib.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// ===== WiFi Config =====
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";

// ===== Backend API =====
const char* API_BASE = "http://192.168.1.100:8000"; // change to your PC IP
const char* INGEST_URL = "/api/ingest";

// ===== OLED (0.96" I2C) =====
#define OLED_WIDTH 128
#define OLED_HEIGHT 64
#define OLED_ADDR 0x3C
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, -1);

// ===== DS3231 RTC =====
RTC_DS3231 rtc;

// ===== Sensors =====
// ADC channels: configure pins as per wiring
const int PIN_CURRENT = 34;   // SCT-013 burden output -> ADC
const int PIN_VOLTAGE = 35;   // ZMPT101B output -> ADC

// Calibration
// Adjust these after calibration with a multimeter/load
float currentCalibration = 30.0f; // A per ADC_RMS unit (depends on burden resistor & turns)
float voltageCalibration = 260.0f; // V per ADC_RMS unit (depends on ZMPT gain)

// Sampling settings
const int NUM_SAMPLES = 2000; // ~40 cycles at 50Hz if sampling fast

// Utility to compute ADC-centered RMS from AC sensor
float computeACRMS(int pin) {
  // ESP32 ADC raw 0..4095, center around ~2048 for AC coupling
  uint32_t sumSquares = 0;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    int raw = analogRead(pin);
    int centered = raw - 2048; // assume mid-scale center
    sumSquares += (int32_t)centered * (int32_t)centered;
  }
  float meanSquares = (float)sumSquares / (float)NUM_SAMPLES;
  float rmsCounts = sqrtf(meanSquares);
  return rmsCounts; // in ADC counts
}

float countsToAmps(float counts) {
  // Convert ADC RMS counts to Amps using calibration
  // Also scale counts -> volts: counts/2048 is fraction of full-scale
  float fractionFS = counts / 2048.0f;
  return fractionFS * currentCalibration;
}

float countsToVolts(float counts) {
  float fractionFS = counts / 2048.0f;
  return fractionFS * voltageCalibration;
}

void drawOLED(float Vrms, float Irms, float Watts) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("SmartMeter AI");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  display.setCursor(0, 16);
  display.printf("V: %.1f V\n", Vrms);
  display.setCursor(0, 28);
  display.printf("I: %.2f A\n", Irms);
  display.setCursor(0, 40);
  display.printf("P: %.1f W\n", Watts);

  DateTime now = rtc.now();
  display.setCursor(0, 54);
  display.printf("%02d:%02d:%02d", now.hour(), now.minute(), now.second());

  display.display();
}

void ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(250);
  }
}

void postReading(float Vrms, float Irms, float Watts) {
  ensureWiFi();
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  String url = String(API_BASE) + INGEST_URL;
  if (!http.begin(url)) return;
  http.addHeader("Content-Type", "application/json");

  // Simple payload; backend will split appliances and normalize
  DateTime now = rtc.now();
  char ts[25];
  snprintf(ts, sizeof(ts), "%04d-%02d-%02dT%02d:%02d:%02d", now.year(), now.month(), now.day(), now.hour(), now.minute(), now.second());

  String body = "{";
  body += "\"timestamp\":\"" + String(ts) + "\",";
  body += "\"voltage\":" + String(Vrms, 2) + ",";
  body += "\"current\":" + String(Irms, 3) + ",";
  body += "\"total_power\":" + String(Watts, 1);
  body += "}";

  http.POST(body);
  http.end();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  // OLED init
  if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
    // OLED init failure; continue headless
  }
  display.clearDisplay();
  display.display();

  // RTC init
  if (!rtc.begin()) {
    // RTC not found
  }
  if (rtc.lostPower()) {
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }

  // ADC setup
  analogReadResolution(12); // 0..4095
  analogSetAttenuation(ADC_11db);

  // WiFi (optional at boot)
  ensureWiFi();
}

void loop() {
  // 1) Sample sensors
  float iCountsRMS = computeACRMS(PIN_CURRENT);
  float vCountsRMS = computeACRMS(PIN_VOLTAGE);

  // 2) Convert to engineering units
  float Irms = countsToAmps(iCountsRMS);
  float Vrms = countsToVolts(vCountsRMS);
  float Watts = Vrms * Irms;

  // 3) Show on OLED
  drawOLED(Vrms, Irms, Watts);

  // 4) Send to backend
  postReading(Vrms, Irms, Watts);

  // 5) Update every ~2s
  delay(2000);
}
