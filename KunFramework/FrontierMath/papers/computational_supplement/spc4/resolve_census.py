#!/usr/bin/env python3
"""Resolve AKW Table-1 friend pairs (arXiv:2401.06015) to PD codes.

Each table entry = (label, census_name). The census name IS the complement of the
knot as written, chirality included: entry '-K14n14254' has complement 'v2508', so
Manifold('v2508').exterior_to_link() gives -K14n14254 directly. A negated census
name '-v3195' means complement = mirror(v3195) -> take link then .mirror().

Fallback for K-name entries whose census field is an interval/'10' (complement not
in t<=9 census): snappy.Link(DTname) for <=14 crossings; '-' prefix -> .mirror().

Writes one PD file per resolved entry to SPC4/pds/. Output: SPC4/resolved.json
Run: /tmp/sphenv/bin/python resolve_census.py
"""
import json, os, re, sys

OUT = "/mnt/HIVE/FrontierMath/SPC4"
os.makedirs(f"{OUT}/pds", exist_ok=True)

# (id, labelL, censusL, g4L, labelR, censusR, g4R, traces)  -- AKW Table 1, verbatim
PAIRS = [
 ( 1,"K6a1","s912",1,"19nh_78","[10,11]",1,"Cinf"),
 ( 2,"K9n4","-m199",1,"-18nh_23","o9_34801",1,"Cinf"),
 ( 3,"K10n10","-t12200",0,"-16nh_17","-t11532",0,"not C0"),
 ( 4,"K10n10","-t12200",0,"u_17-28","o9_43446",0,"Cinf"),
 ( 5,"K10n13","m201",1,"u_17-23","o9_34818",1,"Cinf"),
 ( 6,"K11n38","-m224",1,"-16nh_9","v3093",1,"Cinf"),
 ( 7,"K11n49","-v3536",0,"K15n103488","-v3423",0,"unknown"),
 ( 8,"K11n49","-v3536",0,"u_17-27","o9_42735",0,"Cinf"),
 ( 9,"K11n116","-t12748",0,"17nh_28","-t12607",0,"Cinf"),
 (10,"K12n121","s239",1,"u_17-25","-t11441",1,"unknown"),
 (11,"K12n200","t09735",1,"u_17-23b","t11748",1,"unknown"),
 (12,"K12n309","-v3195",0,"-K14n14254","v2508",0,"not C0"),
 (13,"K12n318","-o9_40519",0,"-18nh_32","-o9_39433",0,"not C0"),
 (14,"K13n469","v2272",0,"-K13n469","-v2272",0,"Cinf"),
 (15,"K13n572","10",1,"-K15n89587","[10,11]",1,"Cinf"),
 (16,"K13n1021","-t09900",1,"u_17-28b","-o9_34908",1,"unknown"),
 (17,"K13n1021","-t09900",1,"K15n101402","-o9_43876",1,"unknown"),
 (18,"K13n2527","10",1,"-K15n9379","t12270",1,"Cinf"),
 (19,"K13n3158","o9_42515",0,"19nh_40","o9_41909",0,"unknown"),
 (20,"K14n3155","10",1,"-K14n3155","10",1,"unknown"),
 (21,"K14n3155","10",1,"u_16-20","t11462",1,"unknown"),
 (22,"K14n3155","10",1,"u_16-20b","-t11462",1,"unknown"),
 (23,"K14n3611","o9_33568",1,"u_16-25","-o9_27542",1,"Cinf"),
 (24,"K14n5084","-o9_33833",1,"u_23-35","-o9_37547",1,"C0"),
 (25,"K15n19499","o9_37768",0,"K15n153789","10",0,"Cinf"),
 (26,"K15n19499","o9_37768",0,"u_17-26","o9_31828",0,"Cinf"),
 (27,"K15n94464","[10,11]",1,"u_17-24","-o9_40081",1,"Cinf"),
 (28,"K15n101402","-o9_43876",1,"u_17-28c","-o9_34908",1,"unknown"),
 (29,"K15n103488","-v3423",0,"u_17-27b","o9_42735",0,"unknown"),
 (30,"K15n153789","10",0,"u_17-26b","o9_31828",0,"unknown"),
 (31,"16nh_17","t11532",0,"u_17-28d","-o9_43446",0,"not C0"),
 (32,"18nh_2","v0595",1,"u_19-30","t12120",1,"unknown"),
 (33,"18nh_7","v2869",1,"u_16-22","t12388",1,"C0"),
 (34,"18nh_16","-t10974",0,"u_17-32","o9_39967",0,"unknown"),
 (35,"19nh_4","t07281",0,"u_23-42","o9_34949",0,"Cinf"),
 (36,"19nh_4","t07281",0,"u_17-31","o9_39806",0,"unknown"),
 (37,"u_16-20c","t11462",1,"u_16-20d","-t11462",1,"Cinf"),
 (38,"u_17-24b","t11900",1,"u_16-28","-o9_40803",1,"C0"),
 (39,"u_17-31a","o9_22951",None,"u_17-31b","-o9_22951",None,"Cinf"),
 (40,"u_17-31c","o9_39806",0,"u_23-42b","o9_34949",0,"unknown"),
 (41,"u_16-26","o9_41058",None,"u_16-26b","-o9_41058",None,"Cinf"),
]

CENSUS_RE = re.compile(r'^-?[msvt]\d+$|^-?o9_\d+$')
KNAME_RE = re.compile(r'^-?K(\d+)([an]\d+)$')

def resolve(label, census):
    """-> (Link or None, how)"""
    import snappy
    # preferred: census manifold (chirality faithful per table)
    if CENSUS_RE.match(census):
        neg = census.startswith('-')
        M = snappy.Manifold(census[1:] if neg else census)
        L = M.exterior_to_link()
        if neg:
            L = L.mirror()
        return L, "census:" + census
    # fallback: named low-crossing knot
    m = KNAME_RE.match(label)
    if m and int(m.group(1)) <= 14:
        neg = label.startswith('-')
        name = label.lstrip('-').lstrip('K')  # 'K12n309' -> '12n309'
        try:
            L = snappy.Link(name)
        except Exception:
            return None, "linkfail"
        if neg:
            L = L.mirror()
        return L, "linkexteriors:" + name
    return None, "unresolvable"

def pd_string(pd):
    return "PD[" + ", ".join("X[" + ",".join(map(str, x)) + "]" for x in pd) + "]"

def main():
    results = []
    seen = {}
    for pid, lL, cL, g4L, lR, cR, g4R, tr in PAIRS:
        entry = {"pair": pid, "traces": tr,
                 "L": {"label": lL, "census": cL, "g4_paper": g4L},
                 "R": {"label": lR, "census": cR, "g4_paper": g4R}}
        for side in ("L", "R"):
            e = entry[side]
            key = e["label"]
            if key in seen:
                e["pdfile"] = seen[key].get("pdfile")
                e["crossings"] = seen[key].get("crossings")
                e["resolved_by"] = seen[key].get("resolved_by")
                continue
            L, how = resolve(lL if side == "L" else lR, cL if side == "L" else cR)
            if L is None:
                e["pdfile"] = None; e["resolved_by"] = how
                seen[key] = e
                continue
            try:
                L.simplify('global')
            except Exception:
                pass
            pd = L.PD_code()
            fn = f"{OUT}/pds/p{pid:02d}_{side}_{key.replace('-','m')}.txt"
            open(fn, "w").write(pd_string(pd))
            e["pdfile"] = fn; e["crossings"] = len(pd); e["resolved_by"] = how
            seen[key] = e
        results.append(entry)
        print(pid, lL, "->", entry["L"].get("crossings"), "|", lR, "->", entry["R"].get("crossings"), flush=True)
    json.dump(results, open(f"{OUT}/resolved.json", "w"), indent=1)
    res = sum(1 for r in results for s in ("L","R") if r[s].get("pdfile"))
    print("resolved knots:", res)

if __name__ == "__main__":
    main()
