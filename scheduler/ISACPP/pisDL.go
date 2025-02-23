package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"time"

	scv "github.com/NJUPT-ISL/SCV/api/v1"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"

	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/framework"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

const PluginName = "pisdl"

var scheme = runtime.NewScheme()

var flag = false
var currentMgr manager.Manager

type pisdlPlugin struct {
	// Arguments given for the plugin
	pluginArguments framework.Arguments
	Cache	cache.Cache
}

type existingTaskInfo struct {
	Name      string `json:"name"`
	BatchSize string `json:"batchsize"`
	ElapsedIters	string	`json:"elapsediters"`
	TotalEpochs	string	`json:"totalepochs"`
	ItersPerEpoch string `json:"itersperepoch"`
}

type gpuInfo struct {
	Index         string             `json:"index"`
	ExistingTasks []existingTaskInfo `json:"taskinfo"`
}

type transmitData struct {
	Name        string    `json:"name"`
	GPUModel    string    `json:"gpumodel"`
	BatchSize   string    `json:"batchsize"`
	TotalEpochs		string 	`json:"totalepochs"`
	GpuTaskInfo []gpuInfo `json:"gpuinfo"`
}

type workloadMatchData struct {
	BatchSize   string
	TotalEpochs		string
}

type ElapsedIterationsData struct {
	Data string `json:"ELAPSED_ITERATIONS"`
}

type ItersPerEpochData struct {
	Data string `json:"ITERATIONS_PER_EPOCH"`
}

func (pp *pisdlPlugin) Name() string {
	return PluginName
}

func New(arguments framework.Arguments) framework.Plugin {  // `New` is PluginBuilder
	if !flag {
		if err := scv.AddToScheme(scheme); err != nil {
			klog.Error(err)
			return nil
		}

		mgrConfig := ctrl.GetConfigOrDie()
		mgrConfig.QPS = 1000
		mgrConfig.Burst = 1000

		mgr, err := ctrl.NewManager(mgrConfig, ctrl.Options{
			Scheme:             scheme,
			MetricsBindAddress: ":8083",
			LeaderElection:     false,
			Port:               9444,
		})
		if err != nil {
			klog.Error(err)
			return nil
		}
		go func() {
			if err = mgr.Start(ctrl.SetupSignalHandler()); err != nil {
				klog.Error(err)
				panic(err)
			}
		}()

		currentMgr = mgr
		flag = true
	}

	return &pisdlPlugin{
		pluginArguments: arguments,
		Cache: currentMgr.GetCache(),
	}
}

func (pp *pisdlPlugin) OnSessionOpen(ssn *framework.Session) {
	klog.V(4).Infof("Enter pisdl plugin ...")
	if klog.V(4) {
		defer func() {
			klog.V(4).Infof("Leaving pisdl plugin ...")
		}()
	}

	nodeOrderFn := func(task *api.TaskInfo, node *api.NodeInfo) (float64, error) {
		startTime := time.Now()
		currentScv := &scv.Scv{}
		sendData := &transmitData{}

		sendData.Name = task.Pod.Spec.Containers[0].Name

		err := pp.Cache.Get(context.TODO(), types.NamespacedName{Name: node.Name}, currentScv)
		if err != nil {
			klog.Errorf("Get SCV Error: %v", err)
			return 0, err
		}

		sendData.GPUModel = currentScv.Status.CardList[0].Model

		commands := task.Pod.Spec.Containers[0].Args
		regexBatchSize := regexp.MustCompile(`BATCH_SIZE=([\d.]+)`)
		regexEpochs := regexp.MustCompile(`EPOCHS=([\d.]+)`)

		for _, command := range commands {
			matchBatchSize := regexBatchSize.FindStringSubmatch(command)
			matchEpochs := regexEpochs.FindStringSubmatch(command)
			if len(matchBatchSize) > 1 {
				sendData.BatchSize = matchBatchSize[1]
			} else {
				continue
			}

			if len(matchEpochs) > 1 {
				sendData.TotalEpochs = matchEpochs[1]
			} else {
				continue
			}
		}

		gpus := make([]gpuInfo, 0)

		hostIP := ""

		for _, nodeAddress := range node.Node.Status.Addresses {
			if nodeAddress.Type == v1.NodeInternalIP {
				hostIP = nodeAddress.Address
				break
			} else {
				continue
			}
		}

		for index, gpu := range node.GPUDevices {
			existingTask := make([]existingTaskInfo, 0)

			for _, pod := range gpu.PodMap {
				port := pod.Spec.Containers[0].Ports[0]
				hostPort := port.HostPort

				serverBaseURL := "http://" + hostIP + ":" + strconv.FormatInt(int64(hostPort), 10)

				// Define query parameters (if needed)
				elapsedIterationQueryParams := url.Values{}
				elapsedIterationQueryParams.Add("name", "ELAPSED_ITERATIONS")

				// Build the full server URL with query parameters
				serverURL := fmt.Sprintf("%s/get_env_variable?%s", serverBaseURL, elapsedIterationQueryParams.Encode())

				// Make an HTTP GET request to the server
				response, err := http.Get(serverURL)
				if err != nil {
					fmt.Println("Error for ELAPSED_ITERATIONS:", err)
					response.Body.Close()
					continue
				}

				// Define a struct to unmarshal the JSON data
				var elapsedIterationData ElapsedIterationsData

				// Check if the response status code is 200 (OK)
				if response.StatusCode == http.StatusOK {
					// Read the response body
					body, err := ioutil.ReadAll(response.Body)
					if err != nil {
						fmt.Println("Error reading response body for ELAPSED_ITERATIONS:", err)
						response.Body.Close()
						continue
					}

					// Unmarshal the JSON response into the struct
					if err := json.Unmarshal(body, &elapsedIterationData); err != nil {
						fmt.Println("Error unmarshaling JSON for ELAPSED_ITERATIONS:", err)
						response.Body.Close()
						continue
					}

					// Print the message from the JSON response
					fmt.Println("Server Response:", elapsedIterationData.Data)
				} else {
					fmt.Println("Error for ELAPSED_ITERATIONS:", response.Status)
				}

				response.Body.Close()

				// Define query parameters (if needed)
				itersPerEpochQueryParams := url.Values{}
				itersPerEpochQueryParams.Add("name", "ITERATIONS_PER_EPOCH")

				// Build the full server URL with query parameters
				serverURL = fmt.Sprintf("%s/get_env_variable?%s", serverBaseURL, itersPerEpochQueryParams.Encode())

				// Make an HTTP GET request to the server
				response, err = http.Get(serverURL)
				if err != nil {
					fmt.Println("Error for ITERATIONS_PER_EPOCH:", err)
					response.Body.Close()
					continue
				}

				// Define a struct to unmarshal the JSON data
				var itersPerEpochData ItersPerEpochData

				// Check if the response status code is 200 (OK)
				if response.StatusCode == http.StatusOK {
					// Read the response body
					body, err := ioutil.ReadAll(response.Body)
					if err != nil {
						fmt.Println("Error reading response body for ITERATIONS_PER_EPOCH:", err)
						response.Body.Close()
						continue
					}

					// Unmarshal the JSON response into the struct
					if err := json.Unmarshal(body, &itersPerEpochData); err != nil {
						fmt.Println("Error unmarshaling JSON for ITERATIONS_PER_EPOCH:", err)
						response.Body.Close()
						continue
					}

					// Print the message from the JSON response
					fmt.Println("Server Response:", itersPerEpochData.Data)
				} else {
					fmt.Println("Error for ITERATIONS_PER_EPOCH:", response.Status)
				}

				response.Body.Close()

				matchData := &workloadMatchData{}
				podCommands := pod.Spec.Containers[0].Args
				for _, command := range podCommands {
					matchBatchSize := regexBatchSize.FindStringSubmatch(command)
					matchEpochs := regexEpochs.FindStringSubmatch(command)

					if len(matchBatchSize) > 1 {
						matchData.BatchSize = matchBatchSize[1]
					} else {
						continue
					}

					if len(matchEpochs) > 1 {
						matchData.TotalEpochs = matchEpochs[1]
					} else {
						continue
					}
				}

				existingTask = append(existingTask, existingTaskInfo{
					Name:      pod.Spec.Containers[0].Name,
					BatchSize: matchData.BatchSize,
					TotalEpochs: matchData.TotalEpochs,
					ElapsedIters: elapsedIterationData.Data,
					ItersPerEpoch: itersPerEpochData.Data,
				})
			}

			gpus = append(gpus, gpuInfo{
				Index: strconv.Itoa(index),
				ExistingTasks: existingTask,
			})
		}

		sendData.GpuTaskInfo = gpus

		// Convert feature data to JSON
		sendDataJSON, err := json.Marshal(sendData)
		if err != nil {
			fmt.Println("Error encoding feature data to JSON:", err)
			return 0, err
		}

		// Send the feature data to the deep learning model for inference
		response, err := http.Post("http://192.168.1.204:7000/inference", "application/json", bytes.NewBuffer(sendDataJSON))
		if err != nil {
			fmt.Println("Error sending request to the deep learning model:", err)
			return 0, err
		}
		defer response.Body.Close()

		// Parse the inference result
		var inferenceResult map[string]float64
		if err := json.NewDecoder(response.Body).Decode(&inferenceResult); err != nil {
			fmt.Println("Error decoding inference result:", err)
			return 0, err
		}

		klog.V(4).Infof("Successfully receive inference result from node %s: %v", node.Name, inferenceResult)

		minScore := float64(9999999)

		for index := 0; index < len(inferenceResult); index++ {
			gpuInterference, ok := inferenceResult[strconv.Itoa(index)]
			if !ok {
				fmt.Println("Obtain GPU interference failed for GPU:", index)
			}

			gpuScore := 1000000 - 1000 * gpuInterference

			if gpuScore < minScore {
				minScore = gpuScore
			}
		}

		klog.V(4).Infof("pisDL score for Task %s/%s on node %s is: %v", task.Namespace, task.Name, node.Name, minScore)

		endTime := time.Now()
		elapsed := endTime.Sub(startTime)

		klog.V(4).Infof("pisDL scheduling delay for Task %s/%s on node %s is: %v", task.Namespace, task.Name, node.Name, elapsed)

		return minScore, nil
	}

	ssn.AddNodeOrderFn(pp.Name(), nodeOrderFn)
}

func (pp *pisdlPlugin) OnSessionClose(ssn *framework.Session) {}

func main() {
	framework.RegisterPluginBuilder(PluginName, New)
}
