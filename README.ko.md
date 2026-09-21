# MG90S Dog R1

Rhoban `dog_mujoco`와 `dog_urdf`의 **4다리 × 3관절** 구조를 MG90S로 재설계한 검토용 기계 설계다. 두 원본의 관절명 및 연결 구조를 확인하고, MG90S용 형상·질량·관성·관절 제한을 새로 만들었다. 원본 CAD를 단순 축소한 모델이 아니다.

생성 결과는 `out/mg90s_dog`에 있다. 기존 ESP32 quadruped의 `leg_*.step` 및 `sim` 모델과 별개의 설계다.

## 설계 조건

- MG90S 12개. 몸체 쪽 roll 4개, hip pitch 4개, knee pitch 4개.
- 몸체 판 116 × 66 mm, 어깨 수직 거리 37 mm, 옆으로 벌어진 거리 16 mm.
- 위 다리 축간 44 mm, 아래 다리 축에서 발 중심까지 38 mm, 발 반경 5 mm.
- 기본 무릎각 42°. hip 각도는 발이 hip 아래에 오도록 두 링크 길이로 계산한다.
- roll은 바깥쪽 18°, 안쪽 6°로 제한한다. 안쪽으로 과하게 기울이면 반대편 무릎 서보와 간섭하므로 좌우에 부호가 다른 제한을 적용한다.
- 예상 질량은 생성된 `design_summary.json` 및 `bom.csv`를 따른다. 약 0.31 kg 수준의 조건부 추정이다.
- 배터리 30 g, PCA9685 12 g, 제어보드 10 g, IMU 2 g, 배선/레귤레이터 10 g, 체결부 10 g을 포함한다. 추가 적재물은 0 g.
- 인쇄물 유효 밀도 900 kg/m³는 가정이다. 실제 슬라이서의 필라멘트 사용량 및 실측 무게로 교체한다.

제조사 MG90S 사양에서 질량 13.4 g, 전체 외형 22.8 × 12.2 × 28.5 mm, 4.8 V 스톨 토크 약 0.1765 N·m, 60°/0.10 s를 사용했다. **0.06 N·m 설계 한계는 보수적으로 선택한 가정이며 제조사가 보장한 연속 토크가 아니다.** 장착 귀·출력축 위치·혼 구멍 치수는 실물 확인이 필요한 조정 가능한 값이다.

출처: https://towerpro.com.tw/product/mg90s-3/

## Onshape에서 열기

1. 외형 확인은 `onshape/mg90s_dog_assembly.step`를 가져온다. 서 있는 자세로 저장되어 있다.
2. 운동 가능한 Assembly를 구성하려면 **`onshape/mg90s_dog_zero.step`**를 가져오는 편이 쉽다. 몸체 원점이 (0,0,0)이고 모든 관절의 기하학적 기준각이 0°다. 이 자세는 조립 기준이며 보행용 자세가 아니다.
3. STEP의 하위 구조는 `body`, `fl_shoulder`, `fl_thigh`, `fl_shank` 등 13개 강체 그룹이다. 같은 그룹의 부품은 함께 움직이도록 고정한다.
4. `onshape/onshape_mates.csv`의 원점과 축을 따라 12개 Revolute Mate를 만든다. Mate connector의 Z축을 표의 회전축 방향에 맞춘다. 이름은 `dof_fl_1`부터 `dof_br_3`까지 사용한다.
5. 회전 제한과 기본 자세 각도를 CSV에서 적용한다. `*_1`은 roll, `*_2`는 hip pitch, `*_3`은 knee pitch다.
6. IMU frame은 몸체 기준 (0,0,26) mm, X 전방·Y 왼쪽·Z 위쪽이다. `frame_imu`로 추가할 수 있다.

STEP는 형상·이름·배치를 전달한다. **Onshape의 Mate나 원래 스케치/피처 이력을 자동으로 생성하지는 않는다.** 계층 가져오기 결과에 따라 같은 강체의 부품을 Group 또는 Fastened Mate로 묶는다. CAD 모델은 로컬에서 생성했으며 사용자의 Onshape 문서를 직접 수정하지 않았다.

서 있는 STEP에서 바로 mate를 만들면 현재 자세가 0°로 잡힐 수 있다. 이 경우 CSV의 절대각을 그대로 쓰면 안 되므로 zero STEP를 권장한다. `onshape-to-robot`로 다시 내보낼 때도 관절 축·영점·질량을 현재 MJCF와 비교해야 한다. Onshape의 일반 재료 밀도로 MG90S 내부 질량을 자동 추정하지 말고, `design_summary.json`의 강체 질량 및 관성을 별도로 반영한다.

출처: https://onshape-to-robot.readthedocs.io/en/latest/design.html

## 출력과 조립

`onshape/parts`에는 개별 인쇄물 STEP, `onshape/print_stl`에는 mm 단위 STL이 있다. 부품 파일은 강체 좌표를 유지하므로 슬라이서에서 평평한 면을 바닥에 놓는다. 좌우·앞뒤 장착 방향에 따라 미러 부품이 있으며 파일 이름을 유지한다.

먼저 `servo_fit_coupon.step` 또는 동명의 STL로 서보 케이스 여유·귀 구멍 간격·파일럿 홀을 확인한다. MG90S의 실제 동봉 혼을 사용하며, 모델의 원형 혼은 질량·공간을 위한 대체 형상이다. 인쇄된 스플라인이 실제 서보축과 결합한다고 가정하지 않는다. 실제 혼 형태와 반경 6 mm의 체결 구멍이 다르면 `horn_screw_radius` 등을 수정한다.

서보 귀는 프레임에 나사 체결하고 혼은 회전 브래킷에 체결한다. 혼 중앙 나사 접근 구멍을 두었다. 판/프레임과 나사 맞춤을 확인한 뒤 다리 한 개를 조립해 움직임을 확인한다. 현재 설계는 MG90S 자체 출력축 지지를 이용하며, 외부 베어링을 추가한 양단 지지 설계가 아니다. 낙하·충격 내구성 및 혼/파일럿홀 인발 강도는 실물 검증이 필요하다.

전자부품은 특정 판매자 보드의 구멍 패턴 대신 슬롯과 스트랩으로 고정하는 구성이다. 초록색 부품은 PCA9685 62×25×5 mm, 제어보드 50×25×7 mm, IMU 20×16×3 mm 공간 예약 형상이다. 실물 단자·케이블 굽힘 공간도 확인한다. IMU bridge는 중앙 통로의 지지부에 접착/테이프 등으로 고정하는 초안이며, 특정 모듈의 나사 패턴은 미확정이다.

## 하중과 높이

`validation_report.json`의 `load_screen`은 네 발 지지와 대각선 두 발에 체중을 균등 분배하는 조건을 계산한다. 각 관절의 Jacobian 및 CAD에서 합산한 질량·관성을 사용하고 요구 토크에 1.5를 곱해 0.06 N·m와 비교한다. 두 발 지지는 정적으로 안정하다는 뜻이 아니라 보행 중 하중 분담을 보는 계산 조건이다.

무릎각 25°, 42°, 60°, 75°의 높이·토크를 함께 기록한다. **더 낮추려고 다리를 많이 굽히면 무릎 토크가 증가할 수 있다.** 낮은 몸체가 필요하면 `upper_leg`·`lower_leg`를 줄이는 방법과 굽힘각을 바꾸는 방법을 구별해 계산해야 한다. 다리 길이를 줄이면 서보/혼/프레임 간섭을 다시 확인한다.

보고서의 질량 상한은 같은 자세에서 하중을 비례시킨 근사치이며 보장된 적재 능력이 아니다. 배터리·부품 무게가 바뀌면 `parameters.json`을 수정하고 재생성·검증한다. 스톨 토크·열·기어 유격·전압 강하·순간 접촉력은 이 하중 계산만으로 검증되지 않는다.

## 시뮬레이션

프로젝트 루트에서:

```powershell
python design/mg90s_dog/build.py
python design/mg90s_dog/validate.py
python design/mg90s_dog/check_cad.py
python design/mg90s_dog/run.py --seconds 10
python design/mg90s_dog/run.py --viewer
```

- MuJoCo: `dog_mujoco/scene.xml`. 시뮬레이터 내부 단위 m·kg·s, 1 ms step.
- Isaac Sim 등의 URDF 입력: `dog_urdf/robot.urdf`. `assets` 폴더와 함께 유지한다. floating base로 가져오고 articulation root를 body로 설정한다. 직접 Isaac Sim에서 실행한 결과는 아니다.
- URDF의 effort/velocity 값만으로 실제 구동기가 자동 재현되지 않는다. Isaac Sim에서 position drive, 최대 힘 0.06 N·m, 강성 1.8 N·m/rad 및 감쇠 0.035 N·m·s/rad를 초기값으로 설정하고, 50 Hz 명령/속도 제한을 제어기에서 적용한다. import 옵션 및 drive 단위는 사용 중인 Isaac Sim 버전에서 확인한다.

MuJoCo의 직접 position actuator에는 정적 토크 제한과 PD만 있다. `control.py`의 `ServoBridge`를 쓰면 50 Hz 명령, 명령 1주기 지연, 4 rad/s 명령 변화율 제한이 추가된다. 이 명령 변화율은 실제 축속도 또는 토크-속도 특성의 보장이 아니다. 초기 PD·마찰·관절 armature도 실물 식별 전 가정이다.

검증된 설치 버전은 `requirements-tested.txt`에 기록한다.

## Sim-to-real / real-to-sim

MG90S와 PCA9685만으로 실제 관절각이 피드백되는 것은 아니다. 기본 정책 관측 예시는 gyro 3개, accelerometer 3개, 현재/이전 명령 24개로 총 30개다. 정확한 관절각·관절속도·발 접촉·world 위치·완벽한 quaternion은 정책 관측에 넣지 않았다. `ground_truth_orientation` 센서는 시뮬레이션 평가용이다.

실물 각도 영점·회전 부호·PWM pulse 범위·주파수, 지연, 속도, 하중에 따른 처짐, IMU 바이어스와 축 방향을 측정해 반영한다. PCA9685 채널 제안은 FL 0/1/2, FR 3/4/5, BL 6/7/8, BR 9/10/11 순서다. 기존 ESP32 로봇의 채널 배치와 다르다. `hardware_calibration.template.json`의 null을 실측값으로 채운 뒤 사용한다. 모델은 실제 I2C 명령을 전송하지 않는다.

보행 정책은 아직 학습하지 않았다. 실제 전이 전 질량·마찰·토크·지연·IMU 노이즈·영점 오차를 식별하고 학습 시 변화시키는 작업이 남아 있다. 검증 파일의 standing 성공은 보행/실물 전이 성공을 의미하지 않는다.

## 출처 및 한계

- 원본: https://github.com/Rhoban/onshape-to-robot-examples — `source_audit.json`에 commit, Onshape URL, 원본 질량·관절 비교를 기록한다. 원본 MIT 라이선스는 `LICENSE.Rhoban`에 보존했다.
- MuJoCo: https://mujoco.readthedocs.io/en/stable/XMLreference.html
- Isaac Sim URDF: https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/import_urdf.html

현재 결과는 **실행 가능한 시뮬레이션과 Onshape에서 검토 가능한 CAD 설계 R1**이다. 실제 MG90S 맞춤·하드웨어 장착·열·내구성·인쇄 방향에 따른 강도까지 검증한 제작 확정본은 아니다.
