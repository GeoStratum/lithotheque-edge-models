import os
import zipfile
import shutil

output_dir = r"D:\Dev\Lithoteque\aipacks"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Sources sur E:
base_e = r"E:\fine tuning\rock identifier\fine tuning rock model"
v5_tflite_dir = os.path.join(base_e, "MobileNetV5", "tflite")
v4_tflite_dir = os.path.join(base_e, "MobileNetV4", "tflite")
scale_tflite_dir = os.path.join(base_e, "ScaleDetection_MobileNetV4", "tflite")
sesr_tflite_dir = r"D:\Dev\Lithoteque\model IA\sesr_m5\tflite"

v5_qairt_dir = os.path.join(base_e, "MobileNetV5", "QAIRT")
v4_qairt_dir = os.path.join(base_e, "MobileNetV4", "QAIRT")
scale_qairt_dir = os.path.join(base_e, "ScaleDetection_MobileNetV4", "QAIRT")
sesr_qairt_dir = r"D:\Dev\Lithoteque\model IA\sesr_m5\QAIRT"

v5_tensor_dir = os.path.join(base_e, "MobileNetV5", "GoogleTensor")
v4_tensor_dir = os.path.join(base_e, "MobileNetV4", "GoogleTensor")
scale_tensor_dir = os.path.join(base_e, "ScaleDetection_MobileNetV4", "GoogleTensor")
sesr_tensor_dir = r"D:\Dev\Lithoteque\model IA\sesr_m5\GoogleTensor"

targeting_xml = """<config:device-targeting-config
    xmlns:config="http://schemas.android.com/apk/config">
</config:device-targeting-config>"""

def create_aipack(zip_name, pack_folder_name, files_map):
    zip_path = os.path.join(output_dir, zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("device_targeting_configuration.xml", targeting_xml)
        for src_path, dest_rel_in_assets in files_map.items():
            if os.path.exists(src_path):
                # Dans l'aipack standard PAD :
                # {pack_folder_name}/src/main/assets/{dest_rel_in_assets}
                z.write(src_path, f"{pack_folder_name}/src/main/assets/{dest_rel_in_assets}")
    sz = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  [OK] {zip_name} ({sz:.2f} Mo)")

print("=== 1. GENERATION DES 5 AIPACKS GENERIQUES ===")
# 1. Premium
create_aipack(
    "premium_recognition.aipack", "premium_pack",
    {
        os.path.join(v5_tflite_dir, "best_mobilenetv5_rocks_w4a8.tflite"): "rock_model.tflite",
        os.path.join(scale_tflite_dir, "v4_small_scale_int4.tflite"): "scale_model.tflite"
    }
)
# 2. Balanced
create_aipack(
    "balanced_recognition.aipack", "balanced_pack",
    {
        os.path.join(v4_tflite_dir, "rock_model_w4a8_MobileNetV4.tflite"): "rock_model.tflite",
        os.path.join(scale_tflite_dir, "v4_small_scale_int4.tflite"): "scale_model.tflite"
    }
)
# 3. Standard
create_aipack(
    "standard_recognition.aipack", "standard_pack",
    {
        os.path.join(v5_tflite_dir, "best_mobilenetv5_rocks_int8.tflite"): "rock_model.tflite",
        os.path.join(scale_tflite_dir, "v4_small_scale_int8.tflite"): "scale_model.tflite"
    }
)
# 4. Legacy
create_aipack(
    "legacy_recognition.aipack", "legacy_pack",
    {
        os.path.join(v4_tflite_dir, "rock_model_w8a8_MobileNetV4.tflite"): "rock_model.tflite",
        os.path.join(scale_tflite_dir, "v4_small_scale_int8.tflite"): "scale_model.tflite"
    }
)
# 5. Super-Résolution
create_aipack(
    "super_resolution_recognition.aipack", "super_resolution_pack",
    {
        os.path.join(sesr_tflite_dir, "sesr_m5.tflite"): "ai_pack_super_resolution/sesr_m5.tflite"
    }
)

print("\n=== 2. GENERATION DES 12 AIPACKS GOOGLE TENSOR (G1 à G6) ===")
# 100% INT8 TPU : Pas de W4A8, mais un pack Reconnaissance INT8 + un pack SESR AOT par génération G
tensor_gens = ["Tensor_G1", "Tensor_G2", "Tensor_G3", "Tensor_G4", "Tensor_G5", "Tensor_G6"]
for g in tensor_gens:
    g_lower = g.lower()
    # Pack Reconnaissance INT8 (V5 INT8 + Scale INT8 avec métadonnées AOT TPU)
    create_aipack(
        f"recognition_{g_lower}.aipack", f"recognition_{g_lower}_pack",
        {
            os.path.join(v5_tensor_dir, g, "best_mobilenetv5_rocks_int8.tflite"): "rock_model.tflite",
            os.path.join(scale_tensor_dir, g, "v4_small_scale_int8.tflite"): "scale_model.tflite"
        }
    )
    # Pack Super-Résolution SESR AOT TPU
    create_aipack(
        f"super_resolution_{g_lower}.aipack", f"super_resolution_{g_lower}_pack",
        {
            os.path.join(sesr_tensor_dir, g, "sesr_m5.tflite"): "ai_pack_super_resolution/sesr_m5.tflite"
        }
    )

print("\n=== 3. GENERATION DES 21 AIPACKS QUALCOMM SNAPDRAGON ===")
# A. v69 et v73 : INT8 pur (pas de W4A8 matériel) -> Standard INT8, Legacy INT8, SESR (3 packs chacun = 6 packs)
htp_int8_only = ["v69", "v73"]
for htp in htp_int8_only:
    # Standard INT8 (V5 INT8 + Scale INT8)
    create_aipack(
        f"standard_recognition_snapdragon_{htp}.aipack", f"standard_snapdragon_{htp}_pack",
        {
            os.path.join(v5_qairt_dir, htp, "rock_model_w8a8_MobileNetV5.bin"): "rock_model.bin",
            os.path.join(scale_qairt_dir, htp, "v4_small_scale_int8.bin"): "scale_model.bin"
        }
    )
    # Legacy INT8 (V4 INT8 + Scale INT8)
    create_aipack(
        f"legacy_recognition_snapdragon_{htp}.aipack", f"legacy_snapdragon_{htp}_pack",
        {
            os.path.join(v4_qairt_dir, htp, "rock_model_w8a8_MobileNetV4.bin"): "rock_model.bin",
            os.path.join(scale_qairt_dir, htp, "v4_small_scale_int8.bin"): "scale_model.bin"
        }
    )
    # SESR
    create_aipack(
        f"super_resolution_snapdragon_{htp}.aipack", f"super_resolution_snapdragon_{htp}_pack",
        {
            os.path.join(sesr_qairt_dir, htp, "sesr_m5.bin"): "sesr_m5.bin"
        }
    )

# B. v75, v79, v81, v85, v89 : W4A8 pur (NPU INT4 natif) -> Premium W4A8, Balanced W4A8, SESR (3 packs chacun = 15 packs)
htp_w4a8 = ["v75", "v79", "v81", "v85", "v89"]
for htp in htp_w4a8:
    # Premium W4A8 (V5 W4A8 + Scale INT4)
    create_aipack(
        f"premium_recognition_snapdragon_{htp}.aipack", f"premium_snapdragon_{htp}_pack",
        {
            os.path.join(v5_qairt_dir, htp, "rock_model_w4a8_MobileNetV5.bin"): "rock_model.bin",
            os.path.join(scale_qairt_dir, htp, "v4_small_scale_w4a8.bin"): "scale_model.bin"
        }
    )
    # Balanced W4A8 (V4 W4A8 + Scale INT4)
    create_aipack(
        f"balanced_recognition_snapdragon_{htp}.aipack", f"balanced_snapdragon_{htp}_pack",
        {
            os.path.join(v4_qairt_dir, htp, "rock_model_w4a8_MobileNetV4.bin"): "rock_model.bin",
            os.path.join(scale_qairt_dir, htp, "v4_small_scale_w4a8.bin"): "scale_model.bin"
        }
    )
    # SESR
    create_aipack(
        f"super_resolution_snapdragon_{htp}.aipack", f"super_resolution_snapdragon_{htp}_pack",
        {
            os.path.join(sesr_qairt_dir, htp, "sesr_m5.bin"): "sesr_m5.bin"
        }
    )

print("\n=== TOTAL DES AIPACKS GENERES DANS D:\\Dev\\Lithoteque\\aipacks ===")
all_packs = [f for f in os.listdir(output_dir) if f.endswith(".aipack")]
print(f"Nombre total de packs generes : {len(all_packs)} packs.")
