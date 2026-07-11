# 1. Clean, reflection-free open field dataset (no reverberation coloration)
python3 ./generate_dataset.py \
    --input-wav "/home/jason/Project/pyroomacoustics/examples/input_samples/valorant/weapons/Valorant： Stinger - Gaming Sound Effect (HD).wav" \
    --output-dir ./output_open_field \
    --preset open_field \
    --distance 5.0 \
    --crossover-freq 400

# 2. Indoor Room (Small residential room: wood floor, plasterboard ceiling, brick walls)
python3 ./generate_dataset.py \
    --input-wav ../examples/samples/guitar_44k.wav \
    --output-dir ./output_indoor_room \
    --preset indoor_room \
    --distance 3.0 \
    --crossover-freq 250

# 3. Indoor Wood Room (Medium wood-panelled room)
python3 ./generate_dataset.py \
    --input-wav "/home/jason/Project/pyroomacoustics/examples/input_samples/valorant/weapons/Valorant： Ares Gun - Gaming Sound Effect (HD).wav" \
    --output-dir ./output_indoor_wood_room \
    --preset indoor_wood_room \
    --distance 2.0

# 4. Indoor Hall (Large concrete hall)
python3 ./generate_dataset.py \
    --input-wav ../examples/samples/guitar_44k.wav \
    --output-dir ./output_indoor_hall \
    --preset indoor_hall \
    --distance 2.0

# 5. Indoor Warehouse (Highly reflective metallic box)
python3 ./generate_dataset.py \
    --input-wav "/home/jason/Project/pyroomacoustics/examples/input_samples/valorant/weapons/Valorant： Spectre - Gaming Sound Effect (HD).wav" \
    --output-dir ./output_indoor_warehouse \
    --preset indoor_warehouse \
    --distance 5.0 \
    --crossover-freq 250

# 6. Outdoor Street Canyon (Concrete canyon street, open ends)
python3 ./generate_dataset.py \
    --input-wav ../examples/samples/guitar_44k.wav \
    --output-dir ./output_outdoor_street \
    --preset outdoor_street \
    --distance 75.0 \
    --crossover-freq 250

# 7. Outdoor Grass (Open fields/forests)
python3 ./generate_dataset.py \
    --input-wav ../examples/input_samples/cmu_arctic_us_aew_a0001.wav \
    --output-dir ./output_outdoor_grass \
    --preset outdoor_grass \
    --distance 2.0 \
    --crossover-freq 250

# 8. Outdoor Desert (Dry soil / sand ground)
python3 ./generate_dataset.py \
    --input-wav "/home/jason/Project/pyroomacoustics/examples/input_samples/valorant/weapons/Valorant： Spectre - Gaming Sound Effect (HD).wav" \
    --output-dir ./output_outdoor_desert \
    --preset outdoor_desert \
    --distance 5.0 \
    --crossover-freq 400

# 9. Outdoor Snow (Packed snow ground, ice cliff reflections)
python3 ./generate_dataset.py \
    --input-wav "/home/jason/Project/pyroomacoustics/examples/input_samples/valorant/weapons/Valorant： Stinger - Gaming Sound Effect (HD).wav" \
    --output-dir ./output_outdoor_snow \
    --preset outdoor_snow \
    --distance 5.0 \
    --crossover-freq 250

# 10. Corridor (Narrow hallway, reflective walls)
python3 ./generate_dataset.py \
    --input-wav "/home/jason/Project/pyroomacoustics/examples/input_samples/valorant/weapons/Valorant： Classic Pistol - Gaming Sound Effect (HD).wav" \
    --output-dir ./output_corridor \
    --preset corridor \
    --distance 5.0 \
    --crossover-freq 400

# 11. Tunnel (Masonry/brick sewer tunnel)
python3 ./generate_dataset.py \
    --input-wav "/home/jason/Project/pyroomacoustics/examples/input_samples/valorant/weapons/Valorant： Classic Pistol - Gaming Sound Effect (HD).wav" \
    --output-dir ./output_tunnel \
    --preset tunnel \
    --distance 5.0 \
    --crossover-freq 400

# 12. Courtyard (Semi-open stone plaza with lobby BGM mixed at 15dB SNR)
python3 ./generate_dataset.py \
    --input-wav "/home/jason/Project/pyroomacoustics/examples/input_samples/valorant/weapons/Valorant： Ares Gun - Gaming Sound Effect (HD).wav" \
    --output-dir ./output_courtyard_bgm \
    --preset courtyard \
    --distance 5.0 \
    --bgm-wav "/home/jason/Project/pyroomacoustics/examples/input_samples/valorant_lobby_theme.wav" \
    --bgm-snr 15 \
    --crossover-freq 400

# 13. Dynamic Material Override Example (Wood room base layout, overridden with metal, concrete, and glass surfaces)
python3 ./generate_dataset.py \
    --input-wav "/home/jason/Project/pyroomacoustics/examples/input_samples/valorant/weapons/Valorant： Spectre - Gaming Sound Effect (HD).wav" \
    --output-dir ./output_custom_materials \
    --preset indoor_wood_room \
    --distance 3.0 \
    --crossover-freq 400 \
    --wall-material metal \
    --floor-material concrete \
    --ceiling-material glass
python3 ./generate_dataset.py --preset tunnel --distance 3.0 --visualize
python3 ./generate_dataset.py --preset open_field --distance 3.0 --visualize
python3 ./generate_dataset.py --preset open_field --distance 3.0 --visualize --obstacle