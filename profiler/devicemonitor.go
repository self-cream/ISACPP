package main

import (
	"encoding/csv"
	"fmt"
	"github.com/NVIDIA/go-dcgm/pkg/dcgm"
	"log"
	"os"
	"strconv"
	"time"
)

func main() {
	cleanup, err := dcgm.Init(dcgm.Embedded)
	if err != nil {
		log.Panicln(err)
	}
	defer cleanup()

	gpus, err := dcgm.GetSupportedDevices()
	if err != nil {
		log.Panicln(err)
	}

	interval := time.Duration(1000) * time.Millisecond
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	startTime := time.Now()

	for {
		<-ticker.C
		for idx, gpu := range gpus {
			st, err := dcgm.GetDeviceStatus(gpu)
			if err != nil {
				log.Panicln(err)
			}

			// Update gpu util data of all devices
			fileName := "/home/lzj/test-dcgm/device-monitor-data/gpu-util/device-" + strconv.FormatUint(uint64(idx), 10)  + "-gpu-util-data.csv"
			file, err := os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			if err != nil {
				log.Fatalf("Error creating/opening the file: %v", err)
			}

			writer := csv.NewWriter(file)

			var gpuUtilDataWroten []string
			endTime := time.Now()
			elapsed := endTime.Sub(startTime)

			strElapsedTime := fmt.Sprintf("%.2f", elapsed.Seconds())
			strGpuUtil := fmt.Sprintf("%.2f", float64(st.Utilization.GPU))

			gpuUtilDataWroten = append(gpuUtilDataWroten, strElapsedTime)
			gpuUtilDataWroten = append(gpuUtilDataWroten, strGpuUtil)
			err = writer.Write(gpuUtilDataWroten)

			if err != nil {
				log.Fatalf("Error writing gpu util data: %v", err)
			}

			writer.Flush()
			if err := writer.Error(); err != nil {
				log.Fatalf("Error flushing CSV writer: %v", err)
			}

			log.Println("Data appended to", fileName)

			file.Close()

			// Update gpu util mem of all devices
			fileName = "/home/lzj/test-dcgm/device-monitor-data/gpu-mem/device-" + strconv.FormatUint(uint64(idx), 10)  + "-gpu-mem-data.csv"
			file, err = os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			if err != nil {
				log.Fatalf("Error creating/opening the file: %v", err)
			}

			writer = csv.NewWriter(file)

			var gpuMemDataWroten []string

			endTime = time.Now()
			elapsed = endTime.Sub(startTime)

			strElapsedTime = fmt.Sprintf("%.2f", elapsed.Seconds())
			strGpuMem := fmt.Sprintf("%.2f", float64(st.Utilization.Memory))

			gpuMemDataWroten = append(gpuMemDataWroten, strElapsedTime)
			gpuMemDataWroten = append(gpuMemDataWroten, strGpuMem)
			err = writer.Write(gpuMemDataWroten)

			if err != nil {
				log.Fatalf("Error writing gpu mem data: %v", err)
			}

			writer.Flush()
			if err := writer.Error(); err != nil {
				log.Fatalf("Error flushing CSV writer: %v", err)
			}

			log.Println("Data appended to", fileName)

			file.Close()

			// Update pcie bandwidth of all devices
			fileName = "/home/lzj/test-dcgm/device-monitor-data/pcie-band/device-" + strconv.FormatUint(uint64(idx), 10)  + "-pcie-band-data.csv"
			file, err = os.OpenFile(fileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			if err != nil {
				log.Fatalf("Error creating/opening the file: %v", err)
			}

			writer = csv.NewWriter(file)

			var pcieBandDataWroten []string

			endTime = time.Now()
			elapsed = endTime.Sub(startTime)

			strElapsedTime = fmt.Sprintf("%.2f", elapsed.Seconds())
			strPcieBand := fmt.Sprintf("%.2f", float64(st.PCI.Throughput.Rx))

			pcieBandDataWroten = append(pcieBandDataWroten, strElapsedTime)
			pcieBandDataWroten = append(pcieBandDataWroten, strPcieBand)
			err = writer.Write(pcieBandDataWroten)

			if err != nil {
				log.Fatalf("Error writing pcie bandwidth data: %v", err)
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

//func main(){
//	ret := nvml.Init()
//	if ret != nvml.SUCCESS {
//		log.Fatalf("Unable to initialize NVML: %v", nvml.ErrorString(ret))
//	}
//	defer func() {
//		ret := nvml.Shutdown()
//		if ret != nvml.SUCCESS {
//			log.Fatalf("Unable to shutdown NVML: %v", nvml.ErrorString(ret))
//		}
//	}()
//
//	// Create CSV file
//	file, err := os.Create("/root/lzj/test-dcgm/device-monitor-data/device_average_utilization.csv")
//	if err != nil {
//		log.Fatalf("Could not create CSV file: %v", err)
//	}
//	defer file.Close()
//
//	writer := csv.NewWriter(file)
//	defer writer.Flush()
//
//	// Write CSV header
//	header := []string{"Timestamp", "Average GPU Utilization (%)", "Average Memory Utilization (%)"}
//	if err := writer.Write(header); err != nil {
//		log.Fatalf("Could not write CSV header: %v", err)
//	}
//
//	startTime := time.Now()
//
//	// Main loop to collect and write GPU utilization data every second
//	for {
//		deviceCount, ret := nvml.DeviceGetCount()
//		if ret != nvml.SUCCESS {
//			log.Fatalf("Could not get device count: %v", err)
//		}
//
//		var totalGPUUtilization, totalMemoryUtilization float64
//
//		// Collect data from each GPU
//		for i := 0; i < deviceCount; i++ {
//			device, ret := nvml.DeviceGetHandleByIndex(i)
//			if ret != nvml.SUCCESS {
//				log.Fatalf("Could not get device handle: %v", err)
//			}
//
//			utilization, ret := device.GetUtilizationRates()
//			if ret != nvml.SUCCESS {
//				log.Fatalf("Could not get GPU utilization: %v", err)
//			}
//
//			totalGPUUtilization += float64(utilization.Gpu)
//			totalMemoryUtilization += float64(utilization.Memory)
//		}
//
//		// Calculate averages
//		avgGPUUtilization := totalGPUUtilization / float64(deviceCount)
//		avgMemoryUtilization := totalMemoryUtilization / float64(deviceCount)
//
//		endTime := time.Now()
//		elapsed := endTime.Sub(startTime)
//
//		// Write data to CSV
//		record := []string{
//			fmt.Sprintf("%.2f", elapsed.Seconds()),
//			fmt.Sprintf("%.2f", avgGPUUtilization),
//			fmt.Sprintf("%.2f", avgMemoryUtilization),
//		}
//
//		if err := writer.Write(record); err != nil {
//			log.Fatalf("Could not write CSV record: %v", err)
//		}
//		writer.Flush()
//
//		// Sleep for one second
//		time.Sleep(1 * time.Second)
//	}
//}
