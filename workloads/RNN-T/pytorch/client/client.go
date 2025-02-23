package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
)

type TotalIterationData struct {
	Data string `json:"TOTAL_ITERATIONS"`
}

type ElapsedIterationData struct {
	Data string `json:"ELAPSED_ITERATIONS"`
}

func main() {
	// Define the server base URL
	serverBaseURL := "http://localhost:7000"

	// Define query parameters (if needed)
	totalIterationQueryParams := url.Values{}
	totalIterationQueryParams.Add("name", "TOTAL_ITERATIONS")

	// Build the full server URL with query parameters
	serverURL := fmt.Sprintf("%s/get_env_variable?%s", serverBaseURL, totalIterationQueryParams.Encode())

	// Make an HTTP GET request to the server
	response, err := http.Get(serverURL)
	if err != nil {
		fmt.Println("Error for TOTAL_ITERATIONS:", err)
		return
	}
	defer response.Body.Close()

	// Check if the response status code is 200 (OK)
	if response.StatusCode == http.StatusOK {
		// Read the response body
		body, err := ioutil.ReadAll(response.Body)
		if err != nil {
			fmt.Println("Error reading response body for TOTAL_ITERATIONS:", err)
			return
		}

		// Define a struct to unmarshal the JSON data
		var totalIterationData TotalIterationData

		// Unmarshal the JSON response into the struct
		if err := json.Unmarshal(body, &totalIterationData); err != nil {
			fmt.Println("Error unmarshaling JSON:", err)
			return
		}

		// Print the message from the JSON response
		fmt.Println("Server Response:", totalIterationData.Data)
	} else {
		fmt.Println("Error:", response.Status)
	}

	response.Body.Close()

	 // Define query parameters (if needed)
        elapsedIterationQueryParams := url.Values{}
        elapsedIterationQueryParams.Add("name", "ELAPSED_ITERATIONS")

	// Build the full server URL with query parameters
	serverURL = fmt.Sprintf("%s/get_env_variable?%s", serverBaseURL, elapsedIterationQueryParams.Encode())

	// Make an HTTP GET request to the server
	response, err = http.Get(serverURL)
	if err != nil {
		fmt.Println("Error for ELAPSED_ITERATIONS:", err)
		return
	}
	defer response.Body.Close()

	// Check if the response status code is 200 (OK)
	if response.StatusCode == http.StatusOK {
		// Read the response body
		body, err := ioutil.ReadAll(response.Body)
		if err != nil {
			fmt.Println("Error reading response body for ELAPSED_ITERATIONS:", err)
			return
		}

		// Define a struct to unmarshal the JSON data
		var elapsedIterationData ElapsedIterationData

		// Unmarshal the JSON response into the struct
		if err := json.Unmarshal(body, &elapsedIterationData); err != nil {
			fmt.Println("Error unmarshaling JSON for ELAPSED_ITERATIONS:", err)
			return
		}

		// Print the message from the JSON response
		fmt.Println("Server Response:", elapsedIterationData.Data)
	} else {
		fmt.Println("Error for ELAPSED_ITERATIONS:", response.Status)
	}

	response.Body.Close()
}

