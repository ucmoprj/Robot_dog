"""Small quadruped PLA calibration stand. Units: mm. Run from any directory."""
from pathlib import Path
import json, math, csv
import cadquery as cq
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / 'output'
for folder in ('parts_step', 'print_stl'):
    (OUT / folder).mkdir(parents=True, exist_ok=True)

P = dict(base_length=180, base_width=140, platform_height=150,
         body_width=66, jaw_clearance=0.3, hole_diameter=3.4,
         nut_across_flats=5.7, nut_depth=2.6)
if (ROOT / 'parameters.json').exists():
    P.update(json.loads((ROOT / 'parameters.json').read_text()))
(ROOT / 'parameters.json').write_text(json.dumps(P, indent=2))
H = P['platform_height']
W = P['body_width'] / 2 + P['jaw_clearance']
assert H in range(130,171,10), 'platform_height must be 130..170 in 10 mm increments'
assert 50 <= P['body_width'] <= 80, 'R1 nominal jaw width range: 50..80 mm'

def box(x,y,z,dx,dy,dz):
    return cq.Workplane('XY').box(dx,dy,dz,centered=False).translate((x,y,z)).val()

def drill(s, x,y,z, length, axis=(0,0,1), diameter=None):
    diameter = P['hole_diameter'] if diameter is None else diameter
    return s.cut(cq.Solid.makeCylinder(diameter/2,length,cq.Vector(x,y,z),cq.Vector(*axis)))

def nut(s,x,y,z,depth=2.6):
    cut=cq.Workplane('XY').polygon(6,P['nut_across_flats']/math.cos(math.pi/6)).extrude(depth).val().translate((x,y,z))
    return s.cut(cut)

def slot(s,x,y,z,length,height):
    cut=cq.Workplane('XY').center(x,y).slot2D(length,3.4,90).extrude(height).val().translate((0,0,z))
    return s.cut(cut)

parts={}
base=box(-P['base_length']/2,-P['base_width']/2,0,P['base_length'],P['base_width'],6)
base=cq.Workplane(obj=base).edges('|Z').fillet(8).val()
for x in (-25,25):
    for y in (-14,14):
        base=nut(drill(base,x,y,-1,8),x,y,0)
parts['base']=base

# Upright: flat plate and broad foot, with two triangular webs on its rear.
mast=box(-32,-25,6,64,50,8).fuse(box(-20,-10,14,40,10,102))
for x in (-18,14):
    rib=cq.Workplane('YZ',origin=(x,0,0)).polyline([(-10,14),(-24,14),(-10,90)]).close().extrude(4).val()
    mast=mast.fuse(rib)
for x in (-25,25):
    for y in (-14,14):
        mast=drill(mast,x,y,5,10)
        mast=drill(mast,x,y,12,3,diameter=6.4)
for z in range(20,111,10):
    mast=drill(mast,0,-11,z,12,axis=(0,1,0))
parts['mast']=mast.clean()

# Moving carriage overlaps the upright face. Two bolts 20 mm apart lock height.
car=box(-14,0.3,H-86,28,10,80).fuse(box(-22,-45,H-6,44,90,6))
# Keep the 4 mm ribs inboard of the slots at x=+/-12. The previous ribs
# at x=-14 and 10 covered the two positive-Y slots from below.
# This leaves 0.5 mm beside a 7 mm swept head/tool envelope, and an
# 8 mm central gap for the height-lock bolt heads.
for x in (-8,4):
    rib=cq.Workplane('YZ',origin=(x,0,0)).polyline([(10.3,H-20),(10.3,H-6),(43,H-6)]).close().extrude(4).val()
    car=car.fuse(rib)
for z in (H-80,H-60,H-40,H-20):
    car=drill(car,0,-0.7,z,12,axis=(0,1,0))
for x in (-12,12):
    for side in (-1,1):
        car=slot(car,x,side*24.5,H-7,28.4,8)
    # At the inner end of the positive-Y slots, the wider bolt head also
    # reaches 1.8 mm into the vertical plate. Relieve that edge from below
    # while retaining the platform underside as the bolt-head bearing face.
    access=cq.Workplane('XY').center(x,24.5).slot2D(32,7,90).extrude(80).val().translate((0,0,H-86))
    car=car.cut(access)
parts['carriage']=car.clean()

# Sweep a 7 mm head/tool envelope along the full 25 mm bolt-center travel.
# Check from below the carriage to the platform underside, plus the shank
# passage through the platform. A through-slot alone does not prove access.
carriage_access_checks=[]
for x in (-12,12):
    for side in (-1,1):
        y=side*24.5
        access=cq.Workplane('XY').center(x,y).slot2D(32,7,90).extrude(80).val().translate((0,0,H-86))
        bore=cq.Workplane('XY').center(x,y).slot2D(28.4,3.4,90).extrude(6).val().translate((0,0,H-6))
        blocked=parts['carriage'].intersect(access).Volume()
        bore_blocked=parts['carriage'].intersect(bore).Volume()
        assert blocked<1e-6 and bore_blocked<1e-6, ('carriage slot access',x,y,blocked,bore_blocked)
        carriage_access_checks.append(dict(x_mm=x,y_mm=y,access_diameter_mm=7,
                                           bolt_center_travel_mm=25,
                                           underside_obstruction_mm3=round(blocked,6),
                                           slot_obstruction_mm3=round(bore_blocked,6)))

# Opposed jaws support the belly plate. Caps positively retain its two edges.
for side,label in ((1,'left'),(-1,'right')):
    jaw=box(-20,W-22,H,40,34,6)
    jaw=jaw.fuse(box(-20,W,H+6,40,12,1.7))
    for x in (-12,12):
        jaw=nut(drill(jaw,x,W-10,H-1,8),x,W-10,H+3.4)
        jaw=nut(drill(jaw,x,W+6,H-1,11),x,W+6,H,depth=4.6)
    # Narrow central tongue avoids the body's cylindrical deck posts at x=+/-15.
    cap=box(-20,W,H+8,40,12,4).fuse(box(-8,W-3,H+8,16,3.1,4))
    for x in (-12,12):
        cap=drill(cap,x,W+6,H+7.3,6)
    if side==-1:
        jaw=jaw.mirror('XZ'); cap=cap.mirror('XZ')
    parts['jaw_'+label]=jaw.clean()
    parts['cap_'+label]=cap.clean()

# Coupon: check M3 clearance, hex nut fit and the 0.3 mm plate clearance first.
coupon=box(0,0,0,35,22,6)
coupon=nut(drill(coupon,9,11,-1,8),9,11,0)
coupon=coupon.cut(box(22,0,2,14,16,2.3))
parts['fit_coupon']=coupon

colors={'base':(0.19,0.23,0.29),'mast':(0.15,0.42,0.63),'carriage':(0.15,0.53,0.69)}
assembly=cq.Assembly(name='PLA_small_robot_stand_R1')
print_report=[]
for name,shape in parts.items():
    assert shape.isValid(), name
    assert len(shape.Solids())==1, (name,len(shape.Solids()))
    cq.exporters.export(shape,str(OUT/'parts_step'/f'{name}.step'))
    # Each STL is oriented for printing. Mast/carriage lie on a side face.
    printable=shape
    if name in ('mast','carriage'):
        printable=printable.rotate((0,0,0),(0,1,0),90)
    if name.startswith('jaw_'):
        printable=printable.rotate((0,0,0),(1,0,0),180)
    b=printable.BoundingBox()
    printable=printable.translate((-b.xmin,-b.ymin,-b.zmin))
    cq.exporters.export(printable,str(OUT/'print_stl'/f'{name}.stl'),tolerance=0.06,angularTolerance=0.15)
    b=printable.BoundingBox()
    dims=[round(b.xlen,2),round(b.ylen,2),round(b.zlen,2)]
    print_report.append(dict(part=name,print_bounds_mm=dims,solid_count=len(shape.Solids()),valid=shape.isValid(),fits_220mm_square=max(dims[:2])<=220))
    if name!='fit_coupon':
        assembly.add(shape,name=name,color=cq.Color(*colors.get(name,(0.92,0.53,0.14))))
assembly.save(str(OUT/'stand_assembly.step'))

# Original CAD is stored in each rigid body's local coordinates. Place it using
# MuJoCo forward kinematics, not bounding boxes or hand-entered visual offsets.
import mujoco
model=mujoco.MjModel.from_xml_path(str(REPO/'dog_mujoco/scene.xml'))
data=mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model,data,0)
data.qpos[2]=(H+6+21)/1000
mujoco.mj_forward(model,data)
components=json.loads((REPO/'cad_validation/components.json').read_text())
robot=[]
for item in components:
    local=cq.importers.importBrep(str(REPO/'cad_validation'/f"{item['name']}.brep")).val()
    bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,item['link'])
    assert bid>=0,item['link']
    robot.append((item,local,bid))

def posed(local,bid):
    r=data.xmat[bid].reshape(3,3); t=data.xpos[bid]*1000
    # Location uses a rigid OCC transform and preserves exact analytic surfaces.
    from OCP.gp import gp_Trsf
    tr=gp_Trsf(); tr.SetValues(*[float(v) for row in range(3) for v in [*r[row],t[row]]])
    return local.moved(cq.Location(tr))

combined=cq.Assembly(name='MG90S_with_calibration_stand_R1')
combined.add(assembly,name='stand')
robot_preview=[]
for item,local,bid in robot:
    shape=posed(local,bid)
    color=(0.12,0.20,0.34) if item['kind']=='servo' else ((0.18,0.65,0.42) if item['kind']=='board' else (0.78,0.80,0.82))
    combined.add(shape,name=item['name'],color=cq.Color(*color))
    robot_preview.append((shape,color))
combined.save(str(OUT/'robot_on_stand.step'))

# Exact solid overlap check: assembled stand, then CAD robot at standing and
# one joint at a time through its range. This is not a continuous swept-volume test.
stand_shapes=[(n,s) for n,s in parts.items() if n!='fit_coupon']
clashes=[]
for i,(a,sa) in enumerate(stand_shapes):
    for b,sb in stand_shapes[i+1:]:
        vol=sa.intersect(sb).Volume()
        if vol>0.05: clashes.append(dict(a=a,b=b,volume_mm3=round(vol,3)))
print('Stand pair intersections:',clashes,flush=True)

def bounds_overlap(a,b):
    return all(getattr(a,k+'max')>getattr(b,k+'min')+1e-6 and getattr(b,k+'max')>getattr(a,k+'min')+1e-6 for k in 'xyz')

stand_bounds=[(n,s,s.BoundingBox()) for n,s in stand_shapes]
qstand=data.qpos.copy()
poses=[('standing',qstand.copy())]
for j in range(12):
    jid=model.actuator_trnid[j,0]
    for k,value in enumerate(np.linspace(*model.jnt_range[jid],5)):
        q=qstand.copy(); q[7+j]=value
        poses.append((f'joint_{j+1}_sample_{k}',q))
robot_clashes=[]
min_z=1e9
for pname,q in poses:
    data.qpos[:]=q; mujoco.mj_forward(model,data)
    for item,local,bid in robot:
        if item['kind']=='allowance': continue
        shape=posed(local,bid); bb=shape.BoundingBox(); min_z=min(min_z,bb.zmin)
        for n,s,sb in stand_bounds:
            if not bounds_overlap(bb,sb): continue
            vol=shape.intersect(s).Volume()
            if vol>0.05:
                robot_clashes.append(dict(pose=pname,robot=item['name'],stand=n,volume_mm3=round(vol,3)))
    if pname=='standing' or pname.endswith('sample_4'): print('Checked',pname,flush=True)
report=dict(parameters=P,parts=print_report,stand_internal_clashes=clashes,
            carriage_slot_fastener_access=carriage_access_checks,
            robot_stand_clashes=robot_clashes,pose_count=len(poses),
            minimum_robot_z_mm=round(min_z,3),
            limitation='Nominal configuration only. Finite one-joint pose samples, not all combined poses. No cables or real fasteners modeled. No physical strength verification.')
(OUT/'validation.json').write_text(json.dumps(report,indent=2))

# Engineering preview from the actual CAD tessellation.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def render(filename,with_robot=True):
    fig=plt.figure(figsize=(11,9)); ax=fig.add_subplot(111,projection='3d')
    all_faces=[]; all_colors=[]
    for s,col in [(s,colors.get(n,(0.92,0.53,0.14))) for n,s in stand_shapes]+(robot_preview if with_robot else []):
        vv,tt=s.tessellate(0.4,0.4)
        vertices=np.array([v.toTuple() for v in vv])
        faces=vertices[np.array(tt)]
        normals=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
        light=np.array([0.4,-0.5,0.75]); light/=np.linalg.norm(light)
        brightness=0.6+0.4*np.abs(normals@light)
        all_faces.extend(faces); all_colors.extend(np.array(col)[None,:]*brightness[:,None])
    ax.add_collection3d(Poly3DCollection(all_faces,facecolor=all_colors,edgecolor='none',alpha=1))
    ax.set(xlim=(-100,100),ylim=(-90,90),zlim=(0,220),xlabel='X / mm',ylabel='Y / mm',zlabel='Z / mm')
    ax.set_box_aspect((200,180,220)); ax.view_init(elev=23,azim=-48)
    ax.set_title('MG90S calibration stand R1 | PLA + M3' if with_robot else 'Adjustable stand | 7 printed parts')
    fig.tight_layout(); fig.savefig(OUT/filename,dpi=160); plt.close(fig)
render('preview_robot.png'); render('preview_stand.png',False)
print(json.dumps(report,indent=2),flush=True)
