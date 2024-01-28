import runpod
import base64
import json
import time
time_out = 120

with open("test_input.json", "r") as f:
    test_input = json.load(f)["input"]

def decode_data(data, save_path):
    fh = open(save_path, "wb")
    fh.write(base64.b64decode(data))
    fh.close()

# Set your API key here
runpod.api_key = ""
assert runpod.api_key != "", "Please set your API key in test_client.py"

# Set your endpoint ID here
endpoint_id = ""
assert endpoint_id != "", "Please set your endpoint ID in test_client.py"
endpoint = runpod.Endpoint(endpoint_id)

# Send the request to the endpoint
print("Sending request...")
run_request = endpoint.run(
    test_input
)
result = run_request
print("Got response!", result.status())

time_waited = 0
if run_request.status() == "IN_QUEUE" or result.status() == "IN_PROGRESS":
    while True:
        print("Waiting for completion... ({}/{})".format(time_waited, time_out))
        time.sleep(5)
        time_waited += 5
        if run_request.status() == "COMPLETED":
            print("Generation completed successfully!")
            print("Here's your image:")
            filename = run_request.output()["filename"]
            save_path = "/tmp/" + filename
            data = run_request.output()["data"]
            decode_data(data, save_path)
            print("Saved to " + save_path)
            break
        elif run_request.status() == "FAILED":
            print("Generation failed after starting :")
            print(run_request._fetch_job()["error"])
            break
        elif time_waited > time_out:
            print("Timeout !")
            break
else:
    print("Generation failed before starting :")
    print(run_request._fetch_job()["error"])
