ml load p7zip
for f in val_set_bjhfy68.zip.*; do
    echo "extracting $f"
    7z x "$f" -p"bhRY5B9zS2" -y
done
unzip validation_labels.zip -p zYX5W7fZ

# password bhRY5B9zS2
# password labels zYX5W7fZ