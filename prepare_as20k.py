import os
import csv
import glob

def prepare_as20k_manifest(audio_dir, csv_file, output_tsv, output_lbl, sample_rate=48000):
    print(f"Processing {csv_file}...")
    
    with open(csv_file, 'r') as f:
        # Skip comment lines starting with #
        lines = [line for line in f if not line.startswith('#')]
    
    reader = csv.reader(lines, skipinitialspace=True)
    
    # Files actually present in audio_dir
    present_files = {os.path.basename(f) for f in glob.glob(os.path.join(audio_dir, "*.wav"))}
    
    tsv_entries = []
    lbl_entries = []
    
    for row in reader:
        if len(row) < 4:
            continue
        ytid = row[0]
        labels = row[3].replace('"', '').strip() # labels is row[3] in standard AudioSet csv
        
        filename = f"{ytid}.wav"
        if filename in present_files:
            # The original manifest used 320000 (10s at 32kHz).
            # The readme says this value doesn't affect results significantly.
            # We'll use 320000 to match the original manifest's format.
            num_samples = 320000
            tsv_entries.append(f"{filename}\t{num_samples}")
            lbl_entries.append(f"{ytid}\t{labels}")
            
    # Write .tsv
    with open(output_tsv, 'w') as f:
        f.write(f"{audio_dir}\n")
        for entry in tsv_entries:
            f.write(f"{entry}\n")
            
    # Write .lbl
    with open(output_lbl, 'w') as f:
        for entry in lbl_entries:
            f.write(f"{entry}\n")
            
    print(f"Generated {output_tsv} and {output_lbl} with {len(tsv_entries)} entries.")

def generate_label_descriptors(input_csv, output_csv):
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # skip header
        
        with open(output_csv, 'w') as out_f:
            for row in reader:
                # EAT expects index,mid,"name"
                # To match exactly, we can use format string or csv.writer with quoting
                index, mid, name = row
                out_f.write(f'{index},{mid},"{name}"\n')
    print(f"Generated {output_csv}")

if __name__ == "__main__":
    base_dir = "/home/adminster/DYH"
    local_as_dir = os.path.join(base_dir, "datasets/audioset")
    output_dir = os.path.join(base_dir, "EAT_manifest/AS20K_local")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Label Descriptors
    generate_label_descriptors(
        os.path.join(local_as_dir, "csv/class_labels_indices.csv"),
        os.path.join(output_dir, "label_descriptors.csv")
    )
    
    # 2. Balanced Train (AS20K)
    prepare_as20k_manifest(
        os.path.join(local_as_dir, "audio/balanced_train_segments"),
        os.path.join(local_as_dir, "csv/balanced_train_segments.csv"),
        os.path.join(output_dir, "train.tsv"),
        os.path.join(output_dir, "train.lbl")
    )
    
    # 3. Eval
    prepare_as20k_manifest(
        os.path.join(local_as_dir, "audio/eval_segments"),
        os.path.join(local_as_dir, "csv/eval_segments.csv"),
        os.path.join(output_dir, "eval.tsv"),
        os.path.join(output_dir, "eval.lbl")
    )
