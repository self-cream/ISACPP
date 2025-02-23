package main

import (
	"encoding/csv"
	"strings"

	//	"flag"
	"fmt"
	"github.com/NVIDIA/go-dcgm/pkg/dcgm"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/shirou/gopsutil/v3/process"
	"log"
	"os"
	"strconv"
	"time"
)

type TaskInfo struct {
	Pid 		uint
	Command 	string
	TaskModel 	string
}

type TaskInfoList []TaskInfo

//var process = flag.Uint("pid", 0, "Provide pid to get this process information.")

// NOTE: The "WatchPidFields()" function must be initially called (as root) BEFORE starting the process to be monitored:
// 1. Run as root, for enabling health watches
//   sudo dcgmi stats -e
// 2. Start process to be monitored
// 3. Run processInfo. This is equivalent to "dcgmi stats --pid ENTERPID -v"
//   go build && ./processInfo -pid PID

func main() {
	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		log.Fatalf("Unable to initialize NVML: %v", nvml.ErrorString(ret))
	}
	defer func() {
		ret := nvml.Shutdown()
		if ret != nvml.SUCCESS {
			log.Fatalf("Unable to shutdown NVML: %v", nvml.ErrorString(ret))
		}
	}()

	cleanup, err := dcgm.Init(dcgm.Embedded)
	if err != nil {
		log.Panicln(err)
	}
	defer cleanup()

	// Request DCGM to start recording stats for GPU process fields
	group, err := dcgm.WatchPidFields()
	if err != nil {
		log.Panicln(err)
	}

	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		log.Fatalf("Unable to get device count: %v", nvml.ErrorString(ret))
	}

	// Before retrieving process stats, wait few seconds for watches to be enabled and collect data
	log.Println("Enabling DCGM watches to start collecting process stats. This may take a few seconds....")
	time.Sleep(10000 * time.Millisecond)

	//flag.Parse()

	interval := time.Duration(1000) * time.Millisecond
	ticker := time.NewTicker(interval)

	startTime := time.Now()


	for {
		<-ticker.C
		var gpuTaskInfoList TaskInfoList

		for di := 0; di < count-1; di++ {
			device, ret := nvml.DeviceGetHandleByIndex(di)
			if ret != nvml.SUCCESS {
				log.Fatalf("Unable to get device at index %d: %v", di, nvml.ErrorString(ret))
			}

			processInfos, ret := device.GetComputeRunningProcesses()
			if ret != nvml.SUCCESS {
				log.Fatalf("Unable to get process info for device at index %d: %v", di, nvml.ErrorString(ret))
			}

			fmt.Printf("Found %d processes on device %d\n", len(processInfos), di)
			for _, processInf := range processInfos {
				_, cmdLine := getProcessIDandCMD(strconv.FormatUint(uint64(processInf.Pid), 10))

				gpuTaskModel := getWorkloadFromCommand(cmdLine)

				gpuTaskInfoList = append(gpuTaskInfoList, TaskInfo{
					Pid: 		uint(processInf.Pid),
					Command: 	cmdLine,
					TaskModel: 	gpuTaskModel,
				})

				pidInfo, err := dcgm.GetProcessInfo(group, uint(processInf.Pid))
				if err != nil {
					log.Panicln(err)
				}

				// Update gpu mem data for Pid
				fileName := "/home/lzj/test-dcgm/gpu-monitor-data/gpu-mem/" + strconv.FormatUint(uint64(processInf.Pid), 10) + "-" + gpuTaskModel + "-gpu-mem-data.csv"
				file, err := os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
				if err != nil {
					log.Fatalf("Error creating/opening the file: %v", err)
				}

				writer := csv.NewWriter(file)

				for _, gpu := range pidInfo {
					var dataWroten []string

					endTime := time.Now()
					elapsed := endTime.Sub(startTime)

					strElapsedTime := fmt.Sprintf("%.2f", elapsed.Seconds())
					strGpuMem := fmt.Sprintf("%.2f", *gpu.ProcessUtilization.MemUtil)

					dataWroten = append(dataWroten, strElapsedTime)
					dataWroten = append(dataWroten, strGpuMem)
					err = writer.Write(dataWroten)

					if err != nil {
						log.Fatalf("Error writing gpumem data: %v", err)
					}
				}

				writer.Flush()
				if err := writer.Error(); err != nil {
					log.Fatalf("Error flushing CSV writer: %v", err)
				}

				log.Println("Data appended to", fileName)

				file.Close()

				// Update PCIe bandwidth data for Pid
				fileName = "/home/lzj/test-dcgm/gpu-monitor-data/pcie-band/" + strconv.FormatUint(uint64(processInf.Pid), 10) + "-" + gpuTaskModel + "-pcie-band-data.csv"
				file, err = os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
				if err != nil {
					log.Fatalf("Error creating/opening the file: %v", err)
				}

				writer = csv.NewWriter(file)

				for _, gpu := range pidInfo {
					var dataWroten []string

					endTime := time.Now()
					elapsed := endTime.Sub(startTime)

					strElapsedTime := fmt.Sprintf("%.2f", elapsed.Seconds())
					strPcieBand := fmt.Sprintf("%.2f", float64(gpu.PCI.Throughput.Rx))

					dataWroten = append(dataWroten, strElapsedTime)
					dataWroten = append(dataWroten, strPcieBand)
					err = writer.Write(dataWroten)

					if err != nil {
						log.Fatalf("Error writing pcie bandwidth data: %v", err)
					}
				}

				writer.Flush()
				if err := writer.Error(); err != nil {
					log.Fatalf("Error flushing CSV writer: %v", err)
				}

				log.Println("Data appended to", fileName)

				file.Close()

				// Update gpu util data for Pid
				fileName = "/home/lzj/test-dcgm/gpu-monitor-data/gpu-util/" + strconv.FormatUint(uint64(processInf.Pid), 10) + "-" + gpuTaskModel + "-gpu-util-data.csv"
				file, err = os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
				if err != nil {
					log.Fatalf("Error creating/opening the file: %v", err)
				}

				writer = csv.NewWriter(file)

				for _, gpu := range pidInfo {
					var dataWroten []string

					endTime := time.Now()
					elapsed := endTime.Sub(startTime)

					strElapsedTime := fmt.Sprintf("%.2f", elapsed.Seconds())
					strGpuUtil := fmt.Sprintf("%.2f", *gpu.ProcessUtilization.SmUtil)

					dataWroten = append(dataWroten, strElapsedTime)
					dataWroten = append(dataWroten, strGpuUtil)
					err = writer.Write(dataWroten)

					if err != nil {
						log.Fatalf("Error writing gpuutil data: %v", err)
					}
				}

				writer.Flush()
				if err := writer.Error(); err != nil {
					log.Fatalf("Error flushing CSV writer: %v", err)
				}

				log.Println("Data appended to", fileName)

				file.Close()

				//t := template.Must(template.New("Process").Parse(processInfo))
				//for _, gpu := range pidInfo {
				//
				//	if err = t.Execute(os.Stdout, gpu); err != nil {
				//		log.Panicln("Template error:", err)
				//	}
				//}
			}
		}

		for _, taskInfo := range gpuTaskInfoList {
			var processID int32

			pid := strconv.FormatUint(uint64(taskInfo.Pid), 10)

			pidInt, err := strconv.ParseInt(pid, 10, 32)
			if err != nil {
				fmt.Printf("Error converting string to int32: %v\n", err)
			}

			processList, err := process.Processes()
			if err != nil {
				fmt.Printf("Error fetching process list: %v\n", err)
			}

			for _, process := range processList {
				if int32(pidInt) == process.Pid {
					processID = process.Pid
				}
			}

			if processID == -1 {
				fmt.Println("Process not found.")
				return
			}

			cpuPercent, err := getCPUPercent(processID)
			if err != nil {
				fmt.Printf("Error getting CPU usage: %v", err)
				cpuPercent = 0.0
			}

			// Update cpu util data for Pid
			fileName := "/home/lzj/test-dcgm/gpu-monitor-data/cpu-util/" + strconv.FormatUint(uint64(taskInfo.Pid), 10) + "-" + taskInfo.TaskModel + "-cpu-util-data.csv"
			file, err := os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			if err != nil {
				log.Fatalf("Error creating/opening the file: %v", err)
			}

			writer := csv.NewWriter(file)

			var dataWroten []string

			endTime := time.Now()
			elapsed := endTime.Sub(startTime)

			strElapsedTime := fmt.Sprintf("%.2f", elapsed.Seconds())

			strCpuPercent := fmt.Sprintf("%.2f", cpuPercent)

			dataWroten = append(dataWroten, strElapsedTime)
			dataWroten = append(dataWroten, strCpuPercent)

			err = writer.Write(dataWroten)

			if err != nil {
				log.Fatalf("Error writing gpumem data: %v", err)
			}

			writer.Flush()
			if err := writer.Error(); err != nil {
				log.Fatalf("Error flushing CSV writer: %v", err)
			}

			log.Println("Data appended to", fileName)

			file.Close()
		}
	}
}

func getProcessIDandCMD(pid string) (int32, string) {
	pidInt, err := strconv.ParseInt(pid, 10, 32)
	if err != nil {
		fmt.Printf("Error converting string to int32: %v\n", err)
	}

	processList, err := process.Processes()
	if err != nil {
		fmt.Printf("Error fetching process list: %v\n", err)
		return -1, ""
	}

	for _, process := range processList {
		if int32(pidInt) == process.Pid {
			cmdLine, err := process.Cmdline()
			if err != nil {
				log.Fatal("Error getting command line:", err)
			}
			return process.Pid, cmdLine
		}
	}

	return -1, ""
}

func getCPUPercent(processID int32) (float64, error) {
	//cpuPercent, err := cpu.Percent(time.Second, false)
	//if err != nil {
	//	return 0, err
	//}

	process, err := process.NewProcess(processID)
	if err != nil {
		return 0, err
	}

	processCPUPercent,err := process.Percent(time.Second)

	if err != nil {
		return 0, err
	}

	//processCPUTimes, err := process.Times()
	//if err != nil {
	//	return 0, err
	//}
	//
	//totalCPUPercent := 0.0
	//for _, percent := range cpuPercent {
	//	totalCPUPercent += percent
	//}
	//
	//processCPUPercent := processCPUTimes.Total() / totalCPUPercent * 100.0
	return processCPUPercent, nil
}


func getWorkloadFromCommand(command string) string {
	// Implement logic to extract workload type from the command
	// For example, look for specific arguments that indicate workload type
	if strings.Contains(command, "mask_rcnn") {
		return "Mask R-CNN"
	} else if strings.Contains(command, "unet") {
		return "Unet-3d"
	} else if strings.Contains(command, "single_shot_detector") {
		return "Single-stage-detector"
	} else if strings.Contains(command, "rnn_speech_recognition") {
                return "Rnn-speech-recognition"
        } else if strings.Contains(command, "temporal_fusion_transformer") {
                return "TFT"
        } else if strings.Contains(command, "moflow") {
                return "Gnn-moflow"
        } else if strings.Contains(command, "bert") {
                return "BERT"
        } else if strings.Contains(command, "resnet50") {
                return "Resnet-50"
        } else if strings.Contains(command, "gnmt") {
                return "GNMT"
        } else if strings.Contains(command, "efficientnet_v2") {
                return "Efficientnet-v2"
        } else if strings.Contains(command, "squeezenet") {
                return "Squeezenet"
        } else if strings.Contains(command, "mobilenet") {
		if strings.Contains(command, "mobilenetv2") {
			return "Mobilenetv2"
		}
		return "Mobilenet"
	} else if strings.Contains(command, "shufflenet") {
                if strings.Contains(command, "shufflenetv2") {
                        return "Shufflenetv2"
                }
                return "Shufflenet"
        } else if strings.Contains(command, "vgg11") {
                return "Vgg11"
        } else if strings.Contains(command, "vgg13") {
                return "Vgg13"
        } else if strings.Contains(command, "vgg16") {
                return "Vgg16"
        } else if strings.Contains(command, "vgg19") {
                return "Vgg19"
        } else if strings.Contains(command, "densenet121") {
                return "Densenet121"
        } else if strings.Contains(command, "densenet161") {
                return "Densenet161"
        } else if strings.Contains(command, "densenet201") {
                return "Densenet201"
        } else if strings.Contains(command, "googlenet") {
                return "Googlenet"
        } else if strings.Contains(command, "inceptionv3") {
                return "Inceptionv3"
        } else if strings.Contains(command, "inceptionv4") {
                return "Inceptionv4"
        } else if strings.Contains(command, "inceptionresnetv2") {
                return "Inceptionresnetv2"
        } else if strings.Contains(command, "xception") {
                return "Xception"
        } else if strings.Contains(command, "resnet18") {
                return "Resnet18"
        } else if strings.Contains(command, "resnet34") {
                return "Resnet34"
        } else if strings.Contains(command, "resnet50") {
                return "Resnet50"
        } else if strings.Contains(command, "resnet101") {
                return "Resnet101"
        } else if strings.Contains(command, "resnet152") {
                return "Resnet152"
        } else if strings.Contains(command, "preactresnet18") {
                return "Preactresnet18"
        } else if strings.Contains(command, "preactresnet34") {
                return "Preactresnet34"
        } else if strings.Contains(command, "preactresnet50") {
                return "Preactresnet50"
        } else if strings.Contains(command, "preactresnet101") {
                return "Preactresnet101"
        } else if strings.Contains(command, "preactresnet152") {
                return "Preactresnet152"
        } else if strings.Contains(command, "resnext50") {
                return "Resnext50"
        } else if strings.Contains(command, "resnext101") {
                return "Resnext101"
        } else if strings.Contains(command, "resnext152") {
                return "Resnext152"
        } else if strings.Contains(command, "attention56") {
                return "Attention56"
        } else if strings.Contains(command, "attention92") {
                return "Attention92"
        } else if strings.Contains(command, "seresnet18") {
                return "Seresnet18"
        } else if strings.Contains(command, "seresnet34") {
                return "Seresnet34"
        } else if strings.Contains(command, "seresnet50") {
                return "Seresnet50"
        } else if strings.Contains(command, "seresnet101") {
                return "Seresnet101"
        } else if strings.Contains(command, "seresnet152") {
                return "Seresnet152"
        } else if strings.Contains(command, "nasnet") {
                return "Nasnet"
        } else if strings.Contains(command, "wideresnet") {
                return "Wideresnet"
        } else if strings.Contains(command, "stochasticdepth18") {
                return "Stochasticdepth18"
        } else if strings.Contains(command, "stochasticdepth34") {
                return "Stochasticdepth34"
        } else if strings.Contains(command, "stochasticdepth50") {
                return "Stochasticdepth50"
        } else if strings.Contains(command, "stochasticdepth101") {
                return "Stochasticdepth101"
        }

	return "Unknown" // Default if workload type couldn't be identified
}






