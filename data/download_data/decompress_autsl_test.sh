ml load p7zip
# for f in test_set_xsaft57.zip.*; do
#     echo "extracting $f"
#     7z x "$f" -p"ds6Kvdus3o" -y
    
# done
7z x test_set_xsaft57.zip.001 -p"ds6Kvdus3o" -y
unzip test_labels.zip -p ds6Kvdus3o
# password data: ds6Kvdus3o
# password labels: ds6Kvdus3o