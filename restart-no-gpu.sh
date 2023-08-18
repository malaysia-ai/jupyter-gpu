# when use kubernetes gpu, there is some potential gpu not able to detect.
while true
do
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "No GPU available. Restarting..."
        sleep 5
        reboot
    else
        echo "GPU is available."
    fi
    sleep 60
done