"""Count dataset statistics: patients, MRI slices, MRI masks, reports, QA pairs."""

import json
from pathlib import Path

BASE = Path("/mnt/data/by/project/internvl_2_5")
DATA = Path("/mnt/data/by/data/coco_new/labels_2025_11_29")

PROSPECTIVE_PATIENT_FILTER = {
    "2", "6", "7", "8", "9", "10", "11", "12", "15", "16", "17", "19", "20",
    "21", "22", "23", "24", "26", "27", "31", "32", "33", "35", "37", "39",
    "41", "43", "44", "45", "46", "48", "50", "51", "52", "54", "55", "56",
    "58", "59", "62", "63", "64", "68", "69", "70", "72", "73", "74", "76",
    "77", "78", "79", "80", "81", "82", "83", "84", "86", "88", "92", "93",
    "95", "99", "100", "103", "104", "105", "106", "107", "108", "109", "110",
    "112", "113", "114", "115", "116", "119", "120", "121", "122", "123",
    "124", "126", "127", "128", "129", "130", "131", "132", "133", "134",
    "135", "136", "137", "138", "140", "142", "143", "147",
}


def read_jsonl(path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_patient_from_image(image):
    if isinstance(image, list):
        image = image[0]
    parts = image.split("/")
    if len(parts) >= 4:
        return parts[-4]
    return parts[0]


def get_images(entry):
    img = entry.get("image", "")
    if isinstance(img, list):
        return img
    return [img] if img else []


# ============================================================
# Training data — returns raw sets for aggregation
# ============================================================

def count_training_single():
    qa_files = [
        DATA / "train/train_QA_EN_balanced.jsonl",
        DATA / "train/train_QA_EN_ICC.jsonl",
        DATA / "train/train_QA_EN_Normal.jsonl",
        DATA / "train/train_QA_EN_small_HCC.jsonl",
    ]
    report_files = [
        DATA / "train/train_report_generation_EN.jsonl",
        DATA / "train/train_report_generation_EN_small_HCC.jsonl",
    ]
    mask_files = [
        DATA / "train/train_bbox_v2.jsonl",
        DATA / "train/train_bbox_0_v2.jsonl",
        DATA / "train/train_bbox_small_HCC.jsonl",
    ]

    patients = set()
    images = set()
    mask_imgs = set()
    qa_count = 0
    report_count = 0

    for f in qa_files:
        entries = read_jsonl(f)
        qa_count += len(entries)
        for e in entries:
            patients.add(extract_patient_from_image(e["image"]))
            images.update(get_images(e))

    for f in report_files:
        entries = read_jsonl(f)
        report_count += len(entries)
        for e in entries:
            patients.add(extract_patient_from_image(e["image"]))
            images.update(get_images(e))

    for f in mask_files:
        entries = read_jsonl(f)
        for e in entries:
            patients.add(extract_patient_from_image(e["image"]))
            imgs = get_images(e)
            images.update(imgs)
            mask_imgs.update(imgs)

    return {
        "patients_set": patients, "images_set": images, "mask_images_set": mask_imgs,
        "reports": report_count, "qa": qa_count,
    }


def count_training_multi():
    qa_files = [
        DATA / "multimodal_train/train_QA_EN_multimodal_3mod.jsonl",
        DATA / "multimodal_train/train_QA_EN_multimodal_4mod.jsonl",
        DATA / "multimodal_train/train_QA_EN_small_HCC_multimodal_3mod.jsonl",
        DATA / "multimodal_train/train_QA_EN_small_HCC_multimodal_4mod.jsonl",
        DATA / "multimodal_train/train_QA_EN_ICC_multimodal_4mod.jsonl",
        DATA / "multimodal_train/train_QA_EN_Normal_multimodal_4mod.jsonl",
    ]
    report_files = [
        DATA / "multimodal_train/train_report_generation_EN_multimodal_4mod.jsonl",
        DATA / "multimodal_train/train_report_generation_EN_small_HCC_multimodal_4mod.jsonl",
    ]

    patients = set()
    images = set()
    qa_count = 0
    report_count = 0

    for f in qa_files:
        entries = read_jsonl(f)
        qa_count += len(entries)
        for e in entries:
            patients.add(extract_patient_from_image(e["image"]))
            images.update(get_images(e))

    for f in report_files:
        entries = read_jsonl(f)
        report_count += len(entries)
        for e in entries:
            patients.add(extract_patient_from_image(e["image"]))
            images.update(get_images(e))

    return {
        "patients_set": patients, "images_set": images,
        "reports": report_count, "qa": qa_count,
    }


# ============================================================
# Test data
# ============================================================

SINGLE_DATASETS = [
    ("Internal", "single/internal", "test_QA_EN.json", "test_report_generation_EN.json", ["test_bbox_v2.json", "test_bbox.json"], False),
    ("Ext1 GXMU", "single/external1_gxmu_hcc_icc", "test_QA_EN_external.json", "test_report_generation_EN.json", ["test_bbox.json"], False),
    ("Ext ENSHI", "single/external_enshi", "test_QA_EN.json", "test_report_generation_EN.json", ["test_bbox.json"], False),
    ("Ext GXMU B/N", "single/external_gxmu_benign_normal", "test_QA_EN.json", "test_report_generation_EN.json", ["test_bbox.json"], False),
    ("Ext SanYa", "single/external_sanya", "test_QA_EN.json", "test_report_generation_EN.json", ["test_bbox.json"], False),
    ("Ext Nanning", "single/external_nanning", "test_QA_EN.json", "test_report_generation_EN.json", [], False),
    ("Ext Beihai", "single/external_beihai", "test_QA_EN.json", "test_report_generation_EN.json", [], False),
    ("Ext Guigang", "single/external_guigang", "test_QA_EN.json", "test_report_generation_EN.json", [], False),
    ("Ext Guilin", "single/external_guilin", "test_QA_EN.json", "test_report_generation_EN.json", [], False),
    ("Ext HeNan", "single/external_henan", "test_QA_EN.json", "test_report_generation_EN.json", [], False),
    ("Prospective", "single/prospective_gxmu", "test_QA_EN.json", "test_report_generation_EN.json", ["test_bbox.json"], True),
]

MULTI_DATASETS = [
    ("Internal", "multi_gating/internal", "test_QA_EN_multimodal_4mod.json", "test_report_generation_EN_multimodal.json", False),
    ("Ext1 GXMU", "multi_gating/external1_gxmu_hcc_icc", "test_QA_EN_external_multimodal_4mod.json", "test_report_generation_EN_multimodal.json", False),
    ("Ext ENSHI", "multi_gating/external_enshi", "test_QA_EN_multimodal_4mod.json", "test_report_generation_EN_multimodal.json", False),
    ("Ext GXMU B/N", "multi_gating/external_gxmu_benign_normal", "test_QA_EN_multimodal_4mod.json", "test_report_generation_EN_multimodal.json", False),
    ("Ext SanYa", "multi_gating/external_sanya", "test_QA_EN_multimodal_4mod.json", "test_report_generation_EN_multimodal.json", False),
    ("Ext Nanning", "multi_gating/external_nanning", "test_QA_EN_multimodal_4mod.json", "test_report_generation_EN_multimodal.json", False),
    ("Ext Beihai", "multi_gating/external_beihai", "test_QA_EN_multimodal_4mod.json", "test_report_generation_EN_multimodal.json", False),
    ("Ext Guigang", "multi_gating/external_guigang", "test_QA_EN_multimodal_4mod.json", "test_report_generation_EN_multimodal.json", False),
    ("Ext Guilin", "multi_gating/external_guilin", "test_QA_EN_multimodal_4mod.json", "test_report_generation_EN_multimodal.json", False),
    ("Ext HeNan", "multi_gating/external_henan", "test_QA_EN_multimodal_4mod.json", "test_report_generation_EN_multimodal.json", False),
    ("Prospective", "multi_gating/prospective_gxmu", "test_QA_EN_multimodal_4mod.json", "test_report_generation_EN_multimodal.json", True),
]


def _process_entries(entries, patient_filter, patients_out, images_out):
    """Process entries, applying patient filter if needed. Returns filtered count."""
    count = 0
    for e in entries:
        pid = extract_patient_from_image(e["image"])
        if patient_filter and pid not in patient_filter:
            continue
        count += 1
        patients_out.add(pid)
        images_out.update(get_images(e))
    return count


def count_test_single(ds_dir, qa_file, report_file, grounding_files, use_prospective_filter):
    gt_dir = BASE / "evaluation_archive" / ds_dir / "gt"
    pfilter = PROSPECTIVE_PATIENT_FILTER if use_prospective_filter else None
    patients = set()
    images = set()
    mask_imgs = set()
    qa_count = 0
    report_count = 0

    qa_path = gt_dir / qa_file
    if qa_path.exists():
        qa_count = _process_entries(read_json(qa_path), pfilter, patients, images)

    report_path = gt_dir / report_file
    if report_path.exists():
        report_count = _process_entries(read_json(report_path), pfilter, patients, images)

    for gf in grounding_files:
        gpath = gt_dir / gf
        if gpath.exists():
            for e in read_json(gpath):
                pid = extract_patient_from_image(e["image"])
                if pfilter and pid not in pfilter:
                    continue
                patients.add(pid)
                imgs = get_images(e)
                images.update(imgs)
                mask_imgs.update(imgs)

    return {
        "patients_set": patients, "images_set": images, "mask_images_set": mask_imgs,
        "reports": report_count, "qa": qa_count,
    }


def count_test_multi(ds_dir, qa_file, report_file, use_prospective_filter):
    gt_dir = BASE / "evaluation_archive" / ds_dir / "gt"
    pfilter = PROSPECTIVE_PATIENT_FILTER if use_prospective_filter else None
    patients = set()
    images = set()
    qa_count = 0
    report_count = 0

    qa_path = gt_dir / qa_file
    if qa_path.exists():
        qa_count = _process_entries(read_json(qa_path), pfilter, patients, images)

    report_path = gt_dir / report_file
    if report_path.exists():
        report_count = _process_entries(read_json(report_path), pfilter, patients, images)

    return {
        "patients_set": patients, "images_set": images,
        "reports": report_count, "qa": qa_count,
    }


def print_table(title, rows, headers):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")


def merge_stats_single(stats_list):
    """Merge multiple single-modal stats dicts (union of sets, sum of counts)."""
    patients = set()
    images = set()
    mask_imgs = set()
    reports = 0
    qa = 0
    for s in stats_list:
        patients |= s["patients_set"]
        images |= s["images_set"]
        mask_imgs |= s.get("mask_images_set", set())
        reports += s["reports"]
        qa += s["qa"]
    return {
        "patients_set": patients, "images_set": images, "mask_images_set": mask_imgs,
        "reports": reports, "qa": qa,
    }


def merge_stats_multi(stats_list):
    """Merge multiple multi-modal stats dicts."""
    patients = set()
    images = set()
    reports = 0
    qa = 0
    for s in stats_list:
        patients |= s["patients_set"]
        images |= s["images_set"]
        reports += s["reports"]
        qa += s["qa"]
    return {
        "patients_set": patients, "images_set": images,
        "reports": reports, "qa": qa,
    }


def fmt_single_row(name, s):
    return [name, len(s["patients_set"]), len(s["images_set"]),
            len(s.get("mask_images_set", set())), s["reports"], s["qa"]]


def fmt_multi_row(name, s):
    return [name, len(s["patients_set"]), len(s["images_set"]),
            s["reports"], s["qa"]]


def main():
    # ========== Training ==========
    print("\n*** TRAINING DATA ***")
    s_train = count_training_single()
    m_train = count_training_multi()
    print_table("Training Data", [
        fmt_single_row("Single-modal", s_train),
        [*fmt_multi_row("Multi-modal", m_train)[:3], "-", *fmt_multi_row("Multi-modal", m_train)[3:]],
    ], ["Dataset", "Patients", "MRI slices", "MRI masks", "Reports", "QA pairs"])

    # ========== Test — Single-modal ==========
    print("\n\n*** TEST DATA — SINGLE-MODAL ***")
    single_stats = {}  # name -> raw stats
    single_rows = []
    for name, ds_dir, qa_f, rpt_f, grnd_f, is_prosp in SINGLE_DATASETS:
        stats = count_test_single(ds_dir, qa_f, rpt_f, grnd_f, is_prosp)
        single_stats[name] = stats
        single_rows.append(fmt_single_row(name, stats))

    # External datasets
    ext_names = [n for n, *_ in SINGLE_DATASETS if n.startswith("Ext")]
    ext_all = merge_stats_single([single_stats[n] for n in ext_names])
    # All test
    all_test = merge_stats_single(list(single_stats.values()))
    # All (train + test)
    all_total = merge_stats_single([s_train, all_test])

    single_rows.append(["---"] * 6)
    single_rows.append(fmt_single_row("All External Test", ext_all))
    single_rows.append(fmt_single_row("All Test", all_test))
    single_rows.append(fmt_single_row("All (Train+Test)", all_total))

    print_table("Test — Single-modal", single_rows,
                ["Dataset", "Patients", "MRI slices", "MRI masks", "Reports", "QA pairs"])

    # ========== Test — Multi-modal ==========
    print("\n\n*** TEST DATA — MULTI-MODAL ***")
    multi_stats = {}
    multi_rows = []
    for name, ds_dir, qa_f, rpt_f, is_prosp in MULTI_DATASETS:
        stats = count_test_multi(ds_dir, qa_f, rpt_f, is_prosp)
        multi_stats[name] = stats
        multi_rows.append(fmt_multi_row(name, stats))

    ext_names_m = [n for n, *_ in MULTI_DATASETS if n.startswith("Ext")]
    ext_all_m = merge_stats_multi([multi_stats[n] for n in ext_names_m])
    all_test_m = merge_stats_multi(list(multi_stats.values()))
    all_total_m = merge_stats_multi([m_train, all_test_m])

    multi_rows.append(["---"] * 5)
    multi_rows.append(fmt_multi_row("All External Test", ext_all_m))
    multi_rows.append(fmt_multi_row("All Test", all_test_m))
    multi_rows.append(fmt_multi_row("All (Train+Test)", all_total_m))

    print_table("Test — Multi-modal", multi_rows,
                ["Dataset", "Patients", "MRI slices", "Reports", "QA pairs"])


if __name__ == "__main__":
    main()
