package main

import (
	"fmt"
	"bytes"
	"os/exec"
//	"strings"
)

func main() {
	cmd := exec.Command("docker", "run",
		"-v", "/root/lzj/mlperf/training/rnn_speech_recognition/pytorch:/code",
 		"-v", "/root/lzj/mlperf/training/rnn_speech_recognition/pytorch:/workspace/rnnt",
		"-v", "./datasets/:/datasets",
		"-v", "./checkpoints/:/checkpoints",
		"-v", "./results/:/results",
	        "-i", "--rm", "--gpus", "device=0", "--shm-size=4g", "--ulimit", "memlock=-1", "--ulimit",  "stack=67108864",
		"mlperf/rnn_speech_recognition", "/bin/bash", "-c", "NUM_GPUS=1 GRAD_ACCUMULATION_STEPS=64 GLOBAL_BATCH_SIZE=512 LEARNING_RATE=0.001 EPOCHS=10 bash scripts/train.sh")

        var stdout bytes.Buffer
	var stderr bytes.Buffer

	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		fmt.Println("Error:", err)
		fmt.Println("Standard Error:", stderr.String())
	} else {
		fmt.Println("Standard Output:", stdout.String())
	}
//	err := cmd.Start()
//	if err != nil {
//		fmt.Println("Error:", err)
//		return
//	}
//	fmt.Printf("Pid number: %v\n", int32(cmd.Process.Pid))
}
