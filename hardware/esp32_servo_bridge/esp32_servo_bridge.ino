// ESP32 + PCA9685 servo bridge (minimal firmware for calibration and testing)
//
// The PC (hardware/calibrate.py) sends pulse widths (µs) over USB serial and this passes them
// straight to the PCA9685. All angle conversion and calibration happens on the PC; the firmware stays simple.
//
// Wiring: ESP32 GPIO21 = SDA, GPIO22 = SCL, 3V3 → PCA9685 VCC, common GND.
//         Servo power: a separate 5 V (5 A or more) supply on the PCA9685 V+ terminal. Never power servos from the ESP32's USB.
// Library: install "Adafruit PWM Servo Driver Library" from the Arduino Library Manager.
//
// Serial protocol (115200 baud, one command per line):
//   ?                 → READY
//   P <ch> <us>       → set one channel. us=0 stops the pulse (servo goes limp) → OK
//   M <us0> ... <us11>→ set all 12 channels at once → OK
//   X                 → stop the pulse on every channel → OK
//   invalid command   → ERR <reason>

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

constexpr int SDA_PIN = 21;
constexpr int SCL_PIN = 22;
constexpr int N_CH = 12;
constexpr int US_MIN = 500;   // never output a pulse outside this range
constexpr int US_MAX = 2500;
constexpr uint32_t PCA_OSC_HZ = 27000000;  // varies slightly per board; calibration absorbs the error

Adafruit_PWMServoDriver pca(0x40);
char lineBuf[128];
size_t lineLen = 0;

void setUs(int ch, int us) {
  if (us == 0) {
    pca.setPin(ch, 0);  // full-off: no pulse → servo goes limp and can be turned by hand
    return;
  }
  pca.writeMicroseconds(ch, constrain(us, US_MIN, US_MAX));
}

void allOff() {
  for (int ch = 0; ch < 16; ++ch) pca.setPin(ch, 0);
}

bool parseInt(const char *tok, long &out) {
  if (tok == nullptr) return false;
  char *end;
  out = strtol(tok, &end, 10);
  return *end == '\0';
}

void handleLine(char *line) {
  char *cmd = strtok(line, " ");
  if (cmd == nullptr) return;

  if (strcmp(cmd, "?") == 0) {
    Serial.println("READY");
  } else if (strcmp(cmd, "X") == 0) {
    allOff();
    Serial.println("OK");
  } else if (strcmp(cmd, "P") == 0) {
    long ch, us;
    if (!parseInt(strtok(nullptr, " "), ch) || !parseInt(strtok(nullptr, " "), us) ||
        ch < 0 || ch >= N_CH) {
      Serial.println("ERR usage: P <ch 0-11> <us>");
      return;
    }
    setUs((int)ch, (int)us);
    Serial.println("OK");
  } else if (strcmp(cmd, "M") == 0) {
    long us[N_CH];
    for (int i = 0; i < N_CH; ++i) {
      if (!parseInt(strtok(nullptr, " "), us[i])) {
        Serial.println("ERR usage: M <12 values>");
        return;
      }
    }
    for (int i = 0; i < N_CH; ++i) setUs(i, (int)us[i]);
    Serial.println("OK");
  } else {
    Serial.println("ERR unknown command");
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  pca.begin();
  pca.setOscillatorFrequency(PCA_OSC_HZ);
  pca.setPWMFreq(50);
  allOff();  // no servo moves right after power-up
  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      lineBuf[lineLen] = '\0';
      handleLine(lineBuf);
      lineLen = 0;
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    }
  }
}
