"""Create a separate body revision; all source STL/STEP files remain untouched.

Units: mm. Run: python onshape/build_body_base_v2.py
M3 mounting uses through bolts, washers and nuts, not printed threads.
"""
from pathlib import Path
import hashlib
import json
import cadquery as cq
import trimesh
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / 'body_base_v3'
OUT.mkdir(exist_ok=True)
sources = list(REPO.rglob('*.stl'))
sources = [p for p in sources if OUT not in p.parents]
before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
body = cq.importers.importStep(str(ROOT / 'parts/body_base.step')).val()
original = body
mesh_original = trimesh.load(ROOT / 'print_stl/body_base.stl')
bb = body.BoundingBox()
assert np.allclose(mesh_original.bounds, [[bb.xmin, bb.ymin, bb.zmin],
                                         [bb.xmax, bb.ymax, bb.zmax]], atol=1e-4)
assert abs(mesh_original.volume - body.Volume()) / body.Volume() < 0.001

def box(x, y, z, dx, dy, dz):
    return cq.Solid.makeBox(dx, dy, dz, cq.Vector(x, y, z))

# Downward-open notches rise 2 mm into the underside at z=6.1.
# Offset inward along Y to avoid the original servo screw pilot entirely.
grooves = []
for x in (-42.5, 42.5):
    for y in (-20.5, 20.5):
        cutter = box(x-2.5, y-2, 4.1, 5, 4, 4)
        cutter = cq.Workplane(obj=cutter).edges('|X and >Z').fillet(0.6).val()
        removed = body.intersect(cutter).Volume()
        assert removed > 15
        body = body.cut(cutter)
        assert body.intersect(cutter).Volume() < 1e-6
        # Screw-bearing ring around the original 1.5 mm pilot is preserved.
        pilot_y = 24 if y > 0 else -24
        ring = cq.Solid.makeCylinder(1.5, 3, cq.Vector(x-1.5, pilot_y, 8.3), cq.Vector(1,0,0))
        assert original.intersect(ring).cut(body).Volume() < 1e-6
        # Top bridge remains intact directly above the downward notch.
        top = box(x-1.5, y-2, 10.4, 3, 4, 2)
        assert original.intersect(top).cut(body).Volume() < 1e-6
        grooves.append(dict(x=x, y=y, width=4, depth=2, opening='downward', removed_mm3=removed))

# Two central raised pads: nominal hole centers 20 mm apart along X.
# Each 9.4 x 3.4 slot permits 6 mm bolt-center travel (spacing 14..26).
# PCB hole size and outline remain user-measured, not inferred from ASIN.
for x in (-10.0, 10.0):
    pad = box(x-8, -14, -19.1, 16, 12, 6.1)
    pad = cq.Workplane(obj=pad).edges('|Z').fillet(1.5).val()
    body = body.fuse(pad)
body = body.clean()
for x in (-10.0, 10.0):
    slot = cq.Workplane('XY', origin=(x, -8, -22)).slot2D(9.4, 3.4).extrude(11).val()
    # Open underside recess accepts an M3 nut/washer and allows adjustment.
    recess = cq.Workplane('XY', origin=(x, -8, -21.1)).slot2D(13, 7).extrude(2.9).val()
    body = body.cut(slot).cut(recess)
    assert body.intersect(slot).Volume() < 1e-6
    assert body.intersect(recess).Volume() < 1e-6
body = body.clean()
assert body.isValid() and len(body.Solids()) == 1
assert original.cut(body).Volume() > 0
cq.exporters.export(body, str(OUT / 'body_base_v3.stl'), tolerance=0.04, angularTolerance=0.08)
cq.exporters.export(body, str(OUT / 'body_base_v3.step'))
mesh = trimesh.load(OUT / 'body_base_v3.stl')
assert mesh.is_watertight and mesh.is_winding_consistent
assert len(mesh.split()) == 1
for p, digest in before.items():
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, p

report = dict(units='mm', source='onshape/parts/body_base.step',
              original_stl_files_preserved=len(before), solid_valid=body.isValid(),
              watertight=bool(mesh.is_watertight), connected_components=1,
              grooves=grooves, mounting=dict(slot_width=3.4, slot_length=9.4,
              nominal_spacing=20, adjustable_spacing=[14,26], pcb_bottom_z=-13,
              height_above_base=6, fastener='M3 through bolt + washer + nut',
              underside_recess_width=7, underside_recess_depth=2.8),
              volume_mm3=body.Volume(), original_stl_sha256={str(p.relative_to(REPO)):h for p,h in before.items()})
(OUT / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

fig = plt.figure(figsize=(14,7))
for i, elev, azim in [(1,32,-55),(2,90,-90)]:
    ax = fig.add_subplot(1,2,i,projection='3d')
    colors = np.tile(np.array([0.68,0.77,0.86,1.0]), (len(mesh.faces),1))
    centers = mesh.triangles_center
    colors[(np.abs(centers[:,0])<18.01) & (centers[:,1]>-14.01) & (centers[:,1]<-1.99) & (centers[:,2]>-18.99)] = [0.95,0.63,0.25,1]
    colors[(np.abs(centers[:,0])>40) & (np.abs(centers[:,0])<44.01) & (np.abs(centers[:,1])>18.49) & (np.abs(centers[:,1])<22.51) & (centers[:,2]>6.09) & (centers[:,2]<8.11)] = [0.3,0.75,0.65,1]
    ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolors=colors,edgecolor='#556575',linewidth=0.08))
    ax.set(xlim=(-60,60),ylim=(-36,36),zlim=(-24,14),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)')
    ax.set_box_aspect((120,72,38)); ax.view_init(elev,azim)
    if i == 2:
        ax.set_proj_type('ortho')
        ax.set_zticks([])
        ax.set_zlabel('')
    ax.set_title('Body base v3: downward-open wire notches' if i==1 else 'Top view: M3 slots, 14-26 mm spacing')
fig.tight_layout()
fig.savefig(OUT / 'preview.png', dpi=160)
print(json.dumps({k:v for k,v in report.items() if k!='original_stl_sha256'}, indent=2))
