"""
Collect a clean-vs-attacked camera dataset for the perception-attack detector.

1. Drive a vehicle on autopilot and capture N_CLEAN clean RGB frames (diverse
   street scenes).
2. Offline, apply each of the 5 camera-degradation attacks to every clean frame
   (same params as cameradegration.apply_vision_disruption), giving paired data.
3. Save JPGs under dataset/<label>/ and a manifest.csv (path,label,attack_type).

Usage:  python collect_dataset.py [N_CLEAN]     (default 150)
Run with CARLA up and the venv active (carlasec).
"""
import carla, time, queue, os, sys, csv, random
import numpy as np, cv2

N_CLEAN = int(sys.argv[1]) if len(sys.argv) > 1 else 150
NUM_LOCATIONS = 10     # spread capture across this many spawn points (map areas)
EVERY_K = 4            # keep every 4th frame for scene diversity
IMG_W, IMG_H = 800, 600
DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
ATTACKS = ["blur", "darken", "gaussian_noise", "salt_pepper", "brightness_flicker"]

# ---- degradation (same params as the repo) ----
BLUR_KERNEL=(15,15); DARKEN=0.35; NOISE_STD=35; SP_PROB=0.045; BR_MIN,BR_MAX=1.5,3.5
def degrade(f, t):
    f=f.copy()
    if t=="blur": return cv2.GaussianBlur(f,BLUR_KERNEL,0)
    if t=="darken": return np.clip(f*DARKEN,0,255).astype(np.uint8)
    if t=="gaussian_noise":
        n=np.random.normal(0,NOISE_STD,f.shape).astype(np.float32)
        return np.clip(f.astype(np.float32)+n,0,255).astype(np.uint8)
    if t=="salt_pepper":
        m=np.random.choice((0,1,2),size=f.shape[:2],p=[SP_PROB/2,SP_PROB/2,1-SP_PROB])
        f[m==0]=0; f[m==1]=255; return f
    if t=="brightness_flicker":
        return np.clip(f.astype(np.float32)*random.uniform(BR_MIN,BR_MAX),0,255).astype(np.uint8)
    return f

def main():
    for d in ["clean"] + ATTACKS:
        os.makedirs(os.path.join(DATASET, d), exist_ok=True)

    client = carla.Client("127.0.0.1", 2000); client.set_timeout(20.0)
    world = client.get_world(); bp = world.get_blueprint_library()
    # clean slate
    for a in list(world.get_actors().filter("vehicle.*"))+list(world.get_actors().filter("sensor.*")):
        try: a.destroy()
        except: pass

    spawn_points = world.get_map().get_spawn_points()
    locations = random.sample(spawn_points, min(NUM_LOCATIONS, len(spawn_points)))
    per_loc = max(1, N_CLEAN // len(locations))

    veh = world.spawn_actor(bp.filter("vehicle.*")[0], locations[0])
    veh.set_autopilot(True)
    cbp = bp.find("sensor.camera.rgb"); cbp.set_attribute("image_size_x",str(IMG_W)); cbp.set_attribute("image_size_y",str(IMG_H))
    cam = world.spawn_actor(cbp, carla.Transform(carla.Location(x=1.5,z=1.6)), attach_to=veh)
    q = queue.Queue(); cam.listen(q.put)

    print(f"Collecting ~{N_CLEAN} clean frames across {len(locations)} locations...", flush=True)
    clean_paths = []
    kept = 0
    try:
        for li, loc in enumerate(locations):
            if kept >= N_CLEAN:
                break
            # teleport to a new area and let autopilot resume
            veh.set_autopilot(False); veh.set_transform(loc)
            time.sleep(1.0); veh.set_autopilot(True)
            with q.mutex:  # flush stale frames buffered during the jump
                q.queue.clear()
            time.sleep(1.0)
            i = 0; got = 0
            while got < per_loc and kept < N_CLEAN:
                img = q.get(timeout=15)
                if i % EVERY_K == 0:
                    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
                    bgr = np.ascontiguousarray(arr[:, :, :3])
                    p = os.path.join(DATASET, "clean", f"clean_{kept:04d}.jpg")
                    cv2.imwrite(p, bgr); clean_paths.append(p); kept += 1; got += 1
                i += 1
            print(f"  location {li+1}/{len(locations)}: {kept}/{N_CLEAN} total", flush=True)
    finally:
        cam.destroy(); veh.destroy()
    print(f"Captured {len(clean_paths)} clean frames. Generating attacked versions...", flush=True)

    rows = [(p, "clean", "none") for p in clean_paths]
    for p in clean_paths:
        base = cv2.imread(p); name = os.path.splitext(os.path.basename(p))[0].replace("clean_", "")
        for t in ATTACKS:
            ap = os.path.join(DATASET, t, f"{t}_{name}.jpg")
            cv2.imwrite(ap, degrade(base, t)); rows.append((ap, "attack", t))

    with open(os.path.join(DATASET, "manifest.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path","label","attack_type"]); w.writerows(rows)
    print(f"Done. {len(rows)} images ({len(clean_paths)} clean + {len(clean_paths)*len(ATTACKS)} attacked).", flush=True)
    print(f"Dataset at: {DATASET}", flush=True)

if __name__ == "__main__":
    main()
