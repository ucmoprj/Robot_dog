// ESP32 + PCA9685 + BNO085 bridge (course stage 03: IMU)
//
// Same servo protocol as hardware/esp32_servo_bridge (that sketch is kept unchanged for stages 01-02),
// plus commands to read the BNO085 IMU. The PC does all the math; the firmware only passes data through.
//
// Wiring (both boards on the same I2C bus):
//   ESP32 GPIO21 = SDA → PCA9685 SDA and BNO085 SDA
//   ESP32 GPIO22 = SCL → PCA9685 SCL and BNO085 SCL
//   ESP32 3V3 → PCA9685 VCC and BNO085 VCC (VIN),   common GND for everything
//   BNO085 RST → ESP32 GPIO4 (D4): hardware reset at start and when the watchdog restarts a silent IMU
//   BNO085 INT → ESP32 GPIO27 (D27): LOW = data waiting; the IMU is read only then (+ a 20 ms fallback)
//   BNO085 PS0 and PS1 must be LOW (I2C mode; this is the default on most GY-BNO085 boards)
//   Servo power: a separate 5 V (5 A or more) supply on the PCA9685 V+ terminal.
// Libraries (Arduino Library Manager): "Adafruit PWM Servo Driver Library", "Adafruit BNO08x".
//
// Serial protocol (115200 baud, one command per line):
//   ?                 → READY
//   P <ch> <us>       → set one channel. us=0 stops the pulse (servo goes limp) → OK
//   M <us0> ... <us11>→ set all 12 channels at once → OK
//   X                 → stop the pulse on every channel → OK
//   S                 → IMU status: "IMU OK 0x4A" or "IMU NONE"
//   I                 → latest IMU sample, raw sensor axes:
//                       "I <gx> <gy> <gz> <ax> <ay> <az> <qw> <qx> <qy> <qz> <age_ms>"
//                       gyro in rad/s (calibrated), accel in m/s² INCLUDING gravity (flat and still:
//                       about +9.8 on the axis pointing up), q = game rotation vector (no magnetometer),
//                       age_ms = time since the newest gyro sample. "ERR no imu" if the IMU was not found.
//   R                 → "RESTARTS <n>": how often the watchdog restarted a silent IMU since boot
//   invalid command   → ERR <reason>

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <Adafruit_BNO08x.h>

constexpr int SDA_PIN = 21;
constexpr int SCL_PIN = 22;
constexpr int IMU_RST_PIN = 4;   // BNO085 RST (board label D4); -1 = not connected
constexpr int IMU_INT_PIN = 27;  // BNO085 INT (board label D27); -1 = not connected
constexpr uint32_t INT_FALLBACK_MS = 20;  // read anyway if INT stayed HIGH this long
constexpr int N_CH = 12;
constexpr int US_MIN = 500;   // never output a pulse outside this range
constexpr int US_MAX = 2500;
constexpr uint32_t PCA_OSC_HZ = 27000000;  // varies slightly per board; calibration absorbs the error
constexpr uint32_t IMU_REPORT_US = 10000;  // 100 Hz reports (the policy runs at 50 Hz)
constexpr uint32_t I2C_HZ = 100000;        // 400000 lost the IMU after ~48 s on the bench
constexpr uint32_t IMU_STALL_MS = 1000;    // no gyro report this long → soft-reset the IMU
constexpr uint32_t IMU_RETRY_MS = 2000;    // IMU not found → look again this often; also min time between resets

Adafruit_PWMServoDriver pca(0x40);
Adafruit_BNO08x bno(IMU_RST_PIN);
sh2_SensorValue_t ev;
bool imuOk = false;
uint8_t imuAddr = 0;

float gyro[3] = {0, 0, 0}, accel[3] = {0, 0, 0}, quat[4] = {1, 0, 0, 0};
uint32_t gyroMillis = 0;
uint32_t lastRestart = 0;
uint32_t lastRetry = 0;
uint32_t lastService = 0;
bool shOpen = false;       // the library's link to the IMU is open (must be closed before a new begin)
uint32_t imuRestarts = 0;  // shown by "R"

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

bool enableImuReports() {
  return bno.enableReport(SH2_GYROSCOPE_CALIBRATED, IMU_REPORT_US) &&
         bno.enableReport(SH2_ACCELEROMETER, IMU_REPORT_US) &&
         bno.enableReport(SH2_GAME_ROTATION_VECTOR, IMU_REPORT_US);
}

// Called once for EVERY report the BNO085 sends. The library's getSensorEvent() keeps only the last report
// of each I2C read; the sensor packs reports with the same rate into one read, so the gyro report was
// overwritten every time (seen on the bench: gyro stuck at 0, age_ms = time since boot).
void onSensorEvent(void *, sh2_SensorEvent_t *event) {
  if (sh2_decodeSensorEvent(&ev, event) != SH2_OK) return;
  switch (ev.sensorId) {
    case SH2_GYROSCOPE_CALIBRATED:
      gyro[0] = ev.un.gyroscope.x; gyro[1] = ev.un.gyroscope.y; gyro[2] = ev.un.gyroscope.z;
      gyroMillis = millis();
      break;
    case SH2_ACCELEROMETER:
      accel[0] = ev.un.accelerometer.x; accel[1] = ev.un.accelerometer.y; accel[2] = ev.un.accelerometer.z;
      break;
    case SH2_GAME_ROTATION_VECTOR:
      quat[0] = ev.un.gameRotationVector.real; quat[1] = ev.un.gameRotationVector.i;
      quat[2] = ev.un.gameRotationVector.j; quat[3] = ev.un.gameRotationVector.k;
      break;
  }
}

// (Re)start the IMU. begin_I2C() pulses RST (IMU_RST_PIN), so the chip starts from scratch even if its
// I2C link was stuck. The library has room for ONE open link: a second begin_I2C() without sh2_close()
// fails ("IMU NONE", bench 2026-10-13), so an open link is closed first.
void startImu() {
  if (shOpen) {
    sh2_close();
    shOpen = false;
  }
  // GY-BNO085 boards answer at 0x4A or 0x4B depending on how the address pin is strapped
  for (uint8_t addr : {0x4A, 0x4B}) {
    if (bno.begin_I2C(addr, &Wire)) {
      shOpen = true;
      imuAddr = addr;
      sh2_setSensorCallback(onSensorEvent, nullptr);  // replaces the library's keep-the-last-one handler
      // Bench 2026-10-13: the BNO085 got into a state where it answered every command but enableReport
      // never took effect (report interval stayed 0, no samples), even after power-off. sh2_devOn() (wake
      // from sleep) brought the reports back at once, so it is sent every time.
      sh2_devOn();
      imuOk = enableImuReports();
      gyroMillis = lastRestart = millis();  // the watchdog counts from here, not from boot
      Wire.setClock(I2C_HZ);                // begin() resets the bus speed
      return;
    }
  }
  imuOk = false;
  Wire.setClock(I2C_HZ);
}

// INT goes LOW when the BNO085 has data waiting. Reading only then means the ESP32 does not ask while
// the chip is busy (when it holds SCL low = clock stretching, the likely cause of the stuck link).
// Fallback: also read every INT_FALLBACK_MS, so a missing INT wire only makes it slower, not silent.
bool imuHasData() {
  if (IMU_INT_PIN < 0) return true;
  if (digitalRead(IMU_INT_PIN) == LOW) return true;
  return millis() - lastService > INT_FALLBACK_MS;
}

void pollImu() {
  if (!imuOk) {
    // Not found at boot (wiring, or the IMU powered up late): look again every IMU_RETRY_MS
    if (millis() - lastRetry > IMU_RETRY_MS) {
      lastRetry = millis();
      startImu();
    }
    return;
  }
  if (bno.wasReset()) {  // the BNO085 forgets its reports after a reset
    sh2_devOn();
    enableImuReports();
  }
  if (imuHasData()) {
    lastService = millis();
    sh2_service();  // reads everything waiting; each report goes to onSensorEvent
  }
  // Watchdog: no gyro report for IMU_STALL_MS → hardware reset through RST and start again.
  if (millis() - gyroMillis > IMU_STALL_MS && millis() - lastRestart > IMU_RETRY_MS) {
    ++imuRestarts;
    startImu();
  }
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
  } else if (strcmp(cmd, "S") == 0) {
    if (imuOk) Serial.printf("IMU OK 0x%02X\n", imuAddr);
    else Serial.println("IMU NONE");
  } else if (strcmp(cmd, "I") == 0) {
    if (!imuOk) {
      Serial.println("ERR no imu");
      return;
    }
    Serial.printf("I %.5f %.5f %.5f %.4f %.4f %.4f %.5f %.5f %.5f %.5f %lu\n",
                  gyro[0], gyro[1], gyro[2], accel[0], accel[1], accel[2],
                  quat[0], quat[1], quat[2], quat[3], (unsigned long)(millis() - gyroMillis));
  } else if (strcmp(cmd, "R") == 0) {
    Serial.printf("RESTARTS %lu\n", (unsigned long)imuRestarts);
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
  if (IMU_INT_PIN >= 0) pinMode(IMU_INT_PIN, INPUT_PULLUP);
  startImu();
  Serial.println("READY");
}

void loop() {
  pollImu();
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
