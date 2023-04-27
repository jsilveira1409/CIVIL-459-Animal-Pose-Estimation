import subprocess

cmd = "python3 -m openpifpaf.train \
    --dataset custom_animal \
    --basenet=shufflenetv2k30 \
    --lr=0.00005 \
    --momentum=0.95 \
    --epochs=200 \
    --lr-decay 160 260 \
    --lr-decay-epochs=10  \
    --weight-decay=1e-5 \
    --weight-decay=1e-5 \
    --val-interval 10 \
    --loader-workers 2 \
    --batch-size 1"

result = subprocess.run(cmd, capture_output=True, text=True)

# Print the return code (0 means success)
print("Return code:", result.returncode)

# Print the command output
print("Command output:\n", result.stdout)