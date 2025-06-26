import requests
import json

api_url = "http://127.0.0.1:8000"

# Replace with actual image URLs you can access
front_image_url = "https://cdn.qoneqt.com/uploads/28697/aadhar_front_x6cDEeSvVb.jpeg"
back_image_url = "https://cdn.qoneqt.com/uploads/28697/aadhar_back_79nqM0U7Fv.jpeg"

payload = {
    "user_id": "my_python_test",
    "front_url": front_image_url,
    "back_url": back_image_url,
    "confidence_threshold": 0.6
}

headers = {'Content-Type': 'application/json'}

print(f"Submitting processing request to {api_url}/process...")
try:
    response = requests.post(f"{api_url}/process", data=json.dumps(payload), headers=headers)
    response.raise_for_status() # Raise an exception for bad status codes
    task_info = response.json()
    task_id = task_info['task_id']
    print(f"Task submitted successfully. Task ID: {task_id}")
    print("Initial status:", task_info)

    # Now, poll for status
    print("\nPolling for status...")
    status = "pending"
    while status in ["pending", "processing"]:
        import time
        time.sleep(5) # Wait for 5 seconds before checking again
        status_response = requests.get(f"{api_url}/status/{task_id}")
        status_response.raise_for_status()
        task_info = status_response.json()
        status = task_info['status']
        print(f"Current status: {status} ({task_info.get('message')})")

    if status == "completed":
        print("\nTask completed successfully!")
        print("Processing Time:", task_info.get('processing_time'), "seconds")
        print("Session Directory:", task_info.get('session_dir'))
        print("JSON Results Path (Server-side):", task_info.get('json_results_path'))
        # print("Results:", json.dumps(task_info.get('results'), indent=2)) # Print full results if needed

        # Example of accessing the JSON file via the /results endpoint (if mounted)
        json_filename = "complete_aadhaar_results.json"
        json_url = f"{api_url}/results/{task_info['user_id']}/{task_id}/{json_filename}"
        print(f"Attempting to retrieve JSON via URL: {json_url}")
        try:
            results_file_response = requests.get(json_url)
            results_file_response.raise_for_status()
            print("Successfully retrieved JSON file content:")
            print(json.dumps(results_file_response.json(), indent=2))
        except requests.exceptions.RequestException as e:
             print(f"Failed to retrieve JSON file via /results endpoint: {e}")


    elif status == "error":
        print("\nTask failed!")
        print("Error:", task_info.get('error'))
        print("Failed Step:", task_info.get('failed_step'))
        if task_info.get('security_flagged'):
             print("Security Flagged: Yes")
        # print("Traceback:", task_info.get('traceback')) # Uncomment for debug

except requests.exceptions.RequestException as e:
    print(f"An error occurred during the request: {e}")