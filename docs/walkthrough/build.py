"""Build docs/calibrate_walkthrough.html: a step-by-step, side-by-side tour of
hardware/calibrate.py and hardware/robot_io.py.

The page shows the real source files. Line highlights are found by searching for
anchor text, so small edits to the code do not break them; if an anchor disappears
the build fails and names it. Numbers on the page (pulses, slope, sim range, sweep)
are computed with the same functions the code uses.

Usage (from the project root):
    python docs/walkthrough/build.py
"""
from __future__ import annotations

import builtins
import copy
import html
import io
import json
import keyword
import math
import sys
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HW = ROOT / "hardware"
OUT = ROOT / "docs" / "calibrate_walkthrough.html"

sys.path.insert(0, str(HW))
import calibrate  # noqa: E402
import robot_io  # noqa: E402

SRC = {"cal": (HW / "calibrate.py").read_text(encoding="utf-8"),
       "io": (HW / "robot_io.py").read_text(encoding="utf-8")}
LINES = {k: v.splitlines() for k, v in SRC.items()}


# ---- syntax coloring with Python's own tokenizer ---------------------------------
BUILTINS = set(dir(builtins))


def colorize(src: str) -> list[str]:
    lines = src.splitlines()
    cls = [[None] * len(line) for line in lines]

    def paint(start, end, c):
        (r0, c0), (r1, c1) = start, end
        for r in range(r0, r1 + 1):
            row = cls[r - 1]
            a = c0 if r == r0 else 0
            b = c1 if r == r1 else len(row)
            for i in range(a, min(b, len(row))):
                row[i] = c

    prev = None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        t, s = tok.type, tok.string
        if t == tokenize.COMMENT:
            paint(tok.start, tok.end, "com")
        elif t == tokenize.STRING or t in (getattr(tokenize, "FSTRING_START", -1),
                                           getattr(tokenize, "FSTRING_MIDDLE", -1),
                                           getattr(tokenize, "FSTRING_END", -1)):
            paint(tok.start, tok.end, "str")
        elif t == tokenize.NUMBER:
            paint(tok.start, tok.end, "num")
        elif t == tokenize.NAME:
            if prev is not None and prev.type == tokenize.NAME and prev.string in ("def", "class"):
                paint(tok.start, tok.end, "fn")
            elif prev is not None and prev.type == tokenize.OP and prev.string == "@":
                paint(tok.start, tok.end, "dec")
            elif keyword.iskeyword(s) or s in ("self",):
                paint(tok.start, tok.end, "kw")
            elif s in BUILTINS and not (prev is not None and prev.string == "."):
                paint(tok.start, tok.end, "bi")
        if t not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
            prev = tok

    out = []
    for line, row in zip(lines, cls):
        parts, i = [], 0
        while i < len(line):
            j = i
            while j < len(line) and row[j] == row[i]:
                j += 1
            chunk = html.escape(line[i:j])
            parts.append(f'<span class="t-{row[i]}">{chunk}</span>' if row[i] else chunk)
            i = j
        out.append("".join(parts))
    return out


# ---- line anchors ------------------------------------------------------------------
def ln(key: str, text: str, after: int = 0) -> int:
    for i, line in enumerate(LINES[key][after:], start=after + 1):
        if text in line:
            return i
    raise SystemExit(f"anchor not found in {key}: {text!r}")


def rng(key: str, start: str, end: str | None = None, within: str | None = None) -> list[int]:
    """Lines from `start` to `end`; `within` names a line to search after (for repeated text)."""
    a = ln(key, start, ln(key, within) if within else 0)
    return [a, ln(key, end, a - 1) if end else a]


def block(key: str, header: str) -> list[int]:
    """From a def/class line to the last line of its indented body."""
    a = ln(key, header)
    indent = len(LINES[key][a - 1]) - len(LINES[key][a - 1].lstrip())
    b = a
    for i in range(a, len(LINES[key])):
        line = LINES[key][i]
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        if line.strip():
            b = i + 1
    return [a, b]


# ---- the simulated servo (hidden from the code, like a real one) ---------------------
JOINT = "fl_2"
SPECS = robot_io.load_joint_specs()
SPEC = SPECS[JOINT]
STAND_DEG = math.degrees(SPEC.stand_rad)
TRUE_CENTER, TRUE_DIR, TRUE_K = 1480, -1, 600.0   # standing pose at 1480 µs, 600 µs/rad
HARD = (-62.0, 34.0)                              # degrees where the thigh hits something


def servo_angle(us: int) -> float:
    q = STAND_DEG + math.degrees(TRUE_DIR * (us - TRUE_CENTER) / TRUE_K)
    return max(HARD[0], min(HARD[1], q))


# ---- scenario state ------------------------------------------------------------------
with open(robot_io.TEMPLATE_PATH, encoding="utf-8") as f:
    TEMPLATE = json.load(f)
CH = copy.deepcopy(robot_io.channel_cfg(TEMPLATE, JOINT))

S = {"term": [], "serial": [], "us": None, "q": None, "servo": "off", "esp": "not connected",
     "target": None, "file": "<b>hardware_calibration.json</b>: not loaded yet", "spec": "Not read yet.",
     "marks": {}, "dirty": False, "tested": False}  # marks/dirty/tested mirror the Calibrator attributes
last_cr = False
term_replaced = False
STEPS = []


def fmt_cal(ch: dict) -> dict:
    return {k: (round(ch[k], 4) if k == "zero_rad" and ch[k] is not None else ch[k])
            for k in ("zero_rad", "direction", "center_us", "us_per_rad", "min_us", "max_us")}


def out(line: str) -> None:
    """print(): a line after a \\r echo overwrites that echo, as in a real terminal."""
    global last_cr, term_replaced
    if last_cr:
        S["term"][-1] = line
        term_replaced = True
    else:
        S["term"].append(line)
    last_cr = False


def echo(line: str) -> None:
    """print(..., end='\\r'): keeps rewriting the same terminal line."""
    global last_cr, term_replaced
    if last_cr:
        S["term"][-1] = line
        term_replaced = True
    else:
        S["term"].append(line)
    last_cr = True


def send(us: int, n: int = 1, first: int | None = None) -> None:
    ch = CH["channel"]
    if n == 1:
        S["serial"] += [{"d": "tx", "t": f"P {ch} {us}"}, {"d": "rx", "t": "OK"}]
    else:
        S["serial"] += [{"d": "tx", "t": f"P {ch} {first} … P {ch} {us}", "n": n}, {"d": "rx", "t": "OK", "n": n}]
    S["us"], S["q"], S["servo"] = us, servo_angle(us), "holding"
    S["esp"] = f"channel {ch} → {us} µs (PCA9685 repeats it 50× per second)"


def status() -> None:
    ch = CH
    out(f"  now {S['us']} µs | center={ch['center_us']} dir={ch['direction']} "
        f"us/rad={ch['us_per_rad']} min={ch['min_us']} max={ch['max_us']}"
        f"{'  ✔ done' if robot_io.is_complete(ch) else ''}")
    out(f"  NEXT: {calibrate.next_step_hint(JOINT, ch, SPECS, S['marks'], S['dirty'], S['tested'])}")


def frames_(*names):
    """Call-path labels; an "io:" prefix marks a function in robot_io.py."""
    return [{"n": n.removeprefix("io:"), "io": n.startswith("io:")} for n in names]


def step(kicker, title, text, focus, stack, cal=None, cal2=None, io_=None, io2=None, math_=None, anim=None):
    global term_replaced
    STEPS.append({
        "kicker": kicker, "title": title, "text": text, "math": math_, "focus": focus,
        "stack": frames_(*stack),
        "cal_hit": cal, "cal_hit2": cal2, "io_hit": io_, "io_hit2": io2,
        "term": list(S["term"]), "termReplaced": term_replaced, "serial": list(S["serial"]),
        "us": S["us"], "q": None if S["q"] is None else round(S["q"], 2), "servo": S["servo"],
        "esp": S["esp"], "target": S["target"], "cal": fmt_cal(CH), "file": S["file"], "spec": S["spec"],
        "anim": anim,
    })
    term_replaced = False


# ---- the walkthrough -------------------------------------------------------------------
MAIN = ["main()"]
RUN = ["main()", "Calibrator.run()"]
code = lambda s: f"<code>{html.escape(s)}</code>"  # noqa: E731

S["term"].append(f"> python hardware/calibrate.py --port COM5 --joint {JOINT}")
step("Start", "You run the program",
     f"You type the command. {code('main()')} uses argparse to read the options: which COM port, and which joint to start with. "
     f"Without {code('--port')} or {code('--dry-run')} it stops right here with an error.",
     "cal", MAIN, cal=[rng("cal", "def main", "parser.error")])

S["serial"].append({"d": "sys", "t": "COM5 opened at 115200 baud"})
S["esp"] = "rebooting… then all 16 channels off (setup() calls allOff())"
step("Connect", "Open the USB serial port",
     f"calibrate.py creates a {code('ServoLink')} from robot_io.py. Opening the port reboots the ESP32, so the code waits 2 seconds for it to boot. "
     "The firmware switches every channel off at boot, so nothing moves yet.",
     "io", MAIN + ["io:ServoLink.__init__()"], cal2=[rng("cal", "link = ServoLink")],
     io_=[rng("io", "def __init__(self, port", "reset_input_buffer")])

S["serial"] += [{"d": "tx", "t": "?"}, {"d": "rx", "t": "READY"}]
S["esp"] = "ready · all channels off"
step("Connect", "Handshake: are you there?",
     f"It sends {code('?')} and waits for {code('READY')}. After 5 tries with no answer it stops with "
     f"“No response from the ESP32”. The usual cause: the Arduino Serial Monitor is still open and holding the port.",
     "io", MAIN + ["io:ServoLink.__init__()"], cal2=[rng("cal", "link = ServoLink")],
     io_=[rng("io", "for _ in range(5)", "raise RuntimeError(f\"No response")])

S["file"] = "<b>hardware_calibration.json</b>: not found → started from <b>hardware_calibration.template.json</b> (in memory only)"
step("Load", "Load the calibration table",
     f"{code('Calibrator.__init__')} calls {code('load_calibration()')}. There is no hardware_calibration.json yet, so it copies the template, "
     f"where every value is null. It fills {code('zero_rad')} with the standing-pose angle. The table on the right is fl_2's row.",
     "io", MAIN + ["Calibrator.__init__()", "io:load_calibration()"],
     cal2=[rng("cal", "self.calib = load_calibration()")], io_=[block("io", "def load_calibration")],
     math_=f"zero_rad = standing angle = {STAND_DEG:.1f}° = {SPEC.stand_rad:.4f} rad")

S["spec"] = (f"From onshape_mates.csv: sim range <b>{math.degrees(SPEC.min_rad):.0f}°</b> to <b>+{math.degrees(SPEC.max_rad):.0f}°</b>, "
             f"standing <b>{STAND_DEG:.1f}°</b>. PCA9685 channel <b>{CH['channel']}</b>.")
step("Load", "Read the joint limits from the CSV",
     f"{code('load_joint_specs()')} reads onshape/onshape_mates.csv, the same numbers the simulator uses. "
     "It converts degrees to radians, because the policy works in radians.",
     "io", MAIN + ["Calibrator.__init__()", "io:load_joint_specs()"],
     cal2=[rng("cal", "self.specs = load_joint_specs()")], io_=[block("io", "def load_joint_specs")])

step("Load", "Pick the starting joint",
     f"{code('--joint fl_2')} becomes index 1 in {code('JOINT_ORDER')} (fl_1, fl_2, …, br_3). "
     "The same order is used for the PCA9685 channels and for the policy's 12 outputs, so fl_2 is channel 1 everywhere.",
     "cal", MAIN + ["Calibrator.__init__()"], cal=[rng("cal", "self.idx = JOINT_ORDER.index", "self.marks: dict")],
     io2=[rng("io", "JOINT_ORDER = [")])

for line in (calibrate.HELP + "\n").split("\n")[:-1]:
    out(line)
out("")
for line in calibrate.joint_header(JOINT, robot_io.JOINT_ORDER.index(JOINT), CH, SPECS, 0, False):
    out(line)
step("Select", "Print the key list and what this joint should look like",
     f"{code('run()')} prints the help text, then {code('select()')} prints the joint header from {code('joint_header()')}: where to stand, where to put the phone, "
     "and what the phone should read in pose 1 (standing) and pose 2 (standing + 30° toward the tail).",
     "cal", RUN + ["select()"], cal=[block("cal", "def select")], io2=None, cal2=[rng("cal", "print(HELP)")])

S["target"] = STAND_DEG
step("Select", "First pulse: send() → set_us()",
     f"{code('center_us')} is still null, so {code('select()')} sends 1500 µs, roughly the middle of the servo. "
     f"{code('send()')} keeps it inside 500–2500 µs and calls {code('link.set_us(1, 1500)')} in robot_io.py.",
     "cal", RUN + ["select()", "send()"], cal=[rng("cal", "self.send(ch[\"center_us\"]"), rng("cal", "def send", "self.link.set_us")],
     io2=[block("io", "def set_us")])

send(1500)
step("USB", "_cmd() writes one text line",
     f"{code('set_us()')} formats the line {code('P 1 1500')} and {code('_cmd()')} writes it to USB, then waits for {code('OK')}. "
     "The ESP32 passes it to the PCA9685 over I2C, and the leg jumps to wherever 1500 µs happens to be. Keep a hand near the leg the first time.",
     "io", RUN + ["select()", "send()", "io:set_us()", "io:_cmd()"], cal2=[rng("cal", "self.link.set_us(self.ch")],
     io_=[block("io", "def set_us"), block("io", "def _cmd")],
     math_=f"The phone reads {servo_angle(1500):+.1f}°, the target is {STAND_DEG:.1f}°. Not there yet: this servo's horn sits a little off.")

status()
step("Select", "Print the status line",
     f"{code('status()')} prints the current pulse and the six calibration values (all None: nothing measured yet), "
     f"then a {code('NEXT:')} line from {code('next_step_hint()')} saying what to do next. Every status line comes with one.",
     "cal", RUN + ["select()", "status()"], cal=[rng("cal", "self.status()"), block("cal", "def status")], cal2=[block("cal", "def next_step_hint")])

step("Wait", "Wait for a key",
     f"{code('getch()')} waits for a single key press, without Enter. Nothing happens until you press a key.",
     "cal", RUN, cal=[rng("cal", "while True:", "key = getch()", within="def run")])

send(1490)
echo("  1490 µs")
step("Adjust", "Press a: −10 µs",
     f"Keys a/d/z/c change the pulse by −10/+10/−2/+2 µs. Each press calls {code('send()')} → {code('set_us()')} → one {code('P')} line over USB.",
     "cal", RUN + ["send()"], cal=[rng("cal", "if key in \"adzc\"", "continue")], io2=[block("io", "def set_us")])

send(1480)
echo("  1480 µs")
step("Adjust", "Press a again: on target",
     f"1480 µs. The phone on the thigh now reads {servo_angle(1480):+.1f}°, the standing angle. This is pose 1. "
     "The terminal line is overwritten in place (end=\"\\r\"), so you only see the latest value.",
     "cal", RUN + ["send()"], cal=[rng("cal", "if key in \"adzc\"", "continue")], io2=[block("io", "def set_us")])

CH["center_us"] = 1480
CH["zero_rad"] = SPEC.stand_rad
S["marks"], S["dirty"] = {"center": 1480}, True
out("  center = 1480 µs recorded (in memory; s writes it to the file)")
status()
step("Mark 1", "Press 1: remember the standing pulse",
     f"{code('mark_center()')} stores the current pulse as {code('center_us')}. From now on the code knows: "
     "“1480 µs = standing angle”.",
     "cal", RUN + ["mark_center()"], cal=[rng("cal", "if key == \"1\"", "self.mark_center()"), block("cal", "def mark_center")],
     cal2=[rng("cal", "            self.status()")])

REF_US = 1480 - 310 - 4
send(REF_US, n=33, first=1470)
echo(f"  {REF_US} µs")
S["target"] = STAND_DEG + 30
step("Adjust", "Turn 30° further",
     f"Following the NEXT line (turn the thigh toward the TAIL), you keep pressing until the phone reads "
     f"{STAND_DEG:.1f}° + 30° = {STAND_DEG + 30:+.1f}°. Here that took a ×31 and z ×2. Every press is one P line. "
     "The pulse went <em>down</em>, not up: this servo is mounted so that a + angle needs fewer µs.",
     "cal", RUN + ["send()"], cal=[rng("cal", "if key in \"adzc\"", "continue")])

delta_us = REF_US - CH["center_us"]
delta_rad = math.radians(calibrate.ref_delta_deg(JOINT))
slope = delta_us / delta_rad
CH["direction"] = 1 if slope > 0 else -1
CH["us_per_rad"] = round(abs(slope), 1)
S["marks"]["ref"], S["dirty"] = REF_US, True
out(f"  direction={CH['direction']}, us_per_rad={CH['us_per_rad']} ({abs(slope) * math.pi / 180:.2f} µs/deg)")
lo_us, hi_us = calibrate.expected_delta_us(JOINT)
out(f"  pose 2 is {abs(delta_us)} µs from pose 1. An MG90S usually needs {lo_us}-{hi_us} µs for 30°.")
status()
S["target"] = None
step("Mark 2", "Press 2: direction and µs per radian",
     f"{code('mark_ref()')} compares the two marks. Two points give a straight line: its sign is {code('direction')} and its steepness is "
     f"{code('us_per_rad')}. The pulse moved {abs(REF_US - 1480)} µs, inside the usual "
     f"{calibrate.expected_delta_us(JOINT)[0]}–{calibrate.expected_delta_us(JOINT)[1]} µs for 30°, so no warning.",
     "cal", RUN + ["mark_ref()"], cal=[rng("cal", "elif key == \"2\"", "self.mark_ref()"), block("cal", "def mark_ref")],
     math_=(f"delta_us  = {REF_US} − {CH['center_us']} = {delta_us}\n"
            f"delta_rad = 30° = {delta_rad:.4f} rad\n"
            f"slope     = {delta_us} / {delta_rad:.4f} = {slope:.1f} µs/rad  →  direction = {CH['direction']}, us_per_rad = {CH['us_per_rad']}"))

L3 = REF_US + 71 * 10 + 4 * 2
send(L3, n=75, first=REF_US + 10)
echo(f"  {L3} µs")
step("Limits", "Drive toward one end",
     f"Press d until the thigh is about 4° from touching anything. At {servo_angle(L3):+.1f}° it is close to the front stop (red dashes). "
     "If the servo buzzes, it is already pushing on something: back off at once.",
     "cal", RUN + ["send()"], cal=[rng("cal", "if key in \"adzc\"", "continue")])

S["marks"]["limit3"] = L3
out(f"  limit 3 = {L3} µs recorded")
status()
step("Limits", "Press 3: remember this end",
     f"{code('mark_limit(\"limit3\")')} stores the pulse in {code('self.marks')}. It can't set min/max yet: it needs both ends to know which is which.",
     "cal", RUN + ["mark_limit()"], cal=[rng("cal", "elif key in \"34\"", "self.mark_limit"), rng("cal", "def mark_limit", "print(f\"  limit {key[-1]}")])

L4 = L3 - 92 * 10
send(L4, n=92, first=L3 - 10)
echo(f"  {L4} µs")
step("Limits", "Drive to the other end",
     f"Press a until the thigh is near the back stop: {servo_angle(L4):+.1f}°.",
     "cal", RUN + ["send()"], cal=[rng("cal", "if key in \"adzc\"", "continue")])

lo, hi = sorted((L3, L4))
CH["min_us"], CH["max_us"] = lo, hi
S["marks"]["limit4"] = L4
out(f"  limit 4 = {L4} µs recorded")
out(f"  min_us={lo}, max_us={hi}")
step("Limits", "Press 4: both ends known",
     f"With both marks, {code('mark_limit()')} sorts them into {code('min_us')} and {code('max_us')}. "
     "These become a hard fence: later, no command outside this range will ever be sent to this servo.",
     "cal", RUN + ["mark_limit()"], cal=[block("cal", "def mark_limit")])

ends = sorted(robot_io.rad_to_us_unclamped(CH, q) for q in (SPEC.min_rad, SPEC.max_rad))
out(f"  sim range {ends[0]:.0f}-{ends[1]:.0f} µs fits inside the physical limits ✔")
status()
step("Check", "Does the simulator's range fit?",
     f"{code('check_range()')} converts the sim limits (−48° and +20°) to pulses with the calibration formula in robot_io.py, "
     "and checks that both land inside min_us…max_us. If not, the policy could ask for angles this leg can't reach.",
     "cal", RUN + ["mark_limit()", "check_range()", "io:rad_to_us_unclamped()"],
     cal=[block("cal", "def check_range")], io2=[block("io", "def rad_to_us_unclamped")],
     math_=(f"us = center_us + direction × us_per_rad × (q − zero_rad)\n"
            f"q = −48° → {robot_io.rad_to_us_unclamped(CH, SPEC.min_rad):.0f} µs     q = +20° → {robot_io.rad_to_us_unclamped(CH, SPEC.max_rad):.0f} µs\n"
            f"needed {ends[0]:.0f}–{ends[1]:.0f}  inside  {lo}–{hi}  ✔"))

# sweep, exactly as Calibrator.sweep() steps it
frames = []
dt = 1 / 50
q = SPEC.stand_rad
for target in (SPEC.min_rad, SPEC.max_rad, SPEC.stand_rad):
    while abs(target - q) > 1e-6:
        q += max(-calibrate.SWEEP_SPEED * dt, min(calibrate.SWEEP_SPEED * dt, target - q))
        us = robot_io.rad_to_us(CH, q)
        frames.append([round(servo_angle(us), 2), us])
    frames += [frames[-1]] * 15  # the 0.3 s pause
end_us = robot_io.rad_to_us(CH, SPEC.stand_rad)
n_cmds = sum(1 for f in frames) - 45
S["serial"] += [{"d": "tx", "t": f"P 1 {frames[0][1]} … P 1 {end_us}", "n": n_cmds}, {"d": "rx", "t": "OK", "n": n_cmds}]
S["us"], S["q"], S["servo"] = end_us, servo_angle(end_us), "holding"
S["esp"] = f"channel 1 → {end_us} µs (PCA9685 repeats it 50× per second)"
S["tested"] = True
out("  Sweeping: standing → min → max → standing")
status()
step("Check", "Press t: the code drives the leg",
     f"Now the code moves the leg, not you. Every 20 ms (50 Hz), q steps 0.01 rad (0.57°): standing → −48° → +20° → standing. "
     f"Each q goes through {code('rad_to_us()')}, the same conversion a trained policy's output will go through. "
     "Watch that the leg follows smoothly and never buzzes. (Press Prev/Next to replay.)",
     "cal", RUN + ["sweep()", "io:rad_to_us()"], cal=[block("cal", "def sweep")], io2=[block("io", "def rad_to_us(")],
     math_=f"{n_cmds} P commands in about {n_cmds * dt + 0.9:.1f} s. rad_to_us() also clamps to [{lo}, {hi}] and rounds to whole µs.",
     anim=frames)

S["file"] = ("<b>hardware_calibration.json</b>: written · status <b>PARTIAL_DO_NOT_RUN_POLICY</b> "
             "(1 of 12 joints complete; becomes <b>MEASURED</b> when all 12 are)")
S["dirty"] = False
out(f"  Saved: {robot_io.CALIB_PATH.relative_to(ROOT).as_posix()}")
status()
step("Save", "Press s: write the file",
     f"{code('save_calibration()')} decides the status (MEASURED only when all 12 joints are complete) and writes JSON to disk. "
     "Press s after every joint: if something crashes later, finished joints are safe. Next you would press n for fl_3 and repeat from step 7.",
     "io", RUN + ["io:save_calibration()"], cal2=[rng("cal", "elif key == \"s\"", "print(f\"  Saved:")],
     io_=[block("io", "def save_calibration")])

out("")
out(f"Saved: {robot_io.CALIB_PATH.relative_to(ROOT).as_posix()}  (1/12 joints done)")
step("Quit", "Press q: save again and leave the loop",
     f"q saves once more, counts finished joints and returns from {code('run()')}.",
     "cal", RUN, cal=[rng("cal", "elif key in (\"q\"", "return")])

S["serial"] += [{"d": "tx", "t": "X"}, {"d": "rx", "t": "OK"}, {"d": "sys", "t": "COM5 closed"}]
S["servo"], S["esp"] = "limp", "all channels off · waiting for the next connection"
step("Quit", "Always: relax every servo and close the port",
     f"The {code('finally:')} block runs even after an error or Ctrl+C. {code('close()')} sends {code('X')}, "
     "which stops the pulse on every channel. The leg goes limp and you can move it by hand. Then the port closes.",
     "io", MAIN + ["io:ServoLink.close()", "io:relax_all()"], cal2=[rng("cal", "finally:", "link.close()", within="def main")],
     io_=[block("io", "def relax_all"), block("io", "def close")])

# ---- write the page ------------------------------------------------------------------
data = {
    "files": {"cal": colorize(SRC["cal"]), "io": colorize(SRC["io"])},
    "steps": STEPS,
    "sim": [math.degrees(SPEC.min_rad), math.degrees(SPEC.max_rad)],
    "hard": list(HARD),
}
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
page = (HERE / "template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)
OUT.write_text(page, encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}  ({len(STEPS)} steps)")
