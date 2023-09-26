import runpod
import base64
import json
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
run_request = endpoint.run_sync(
    test_input
)
result = run_request
print("Got response!", result)

# check the status of the request, and if it's completed, save the video
if result["status"] == "COMPLETED":
    print("Generation completed successfully!")
    print("Here's your image:")
    filename = result["output"]["filename"]
    save_path = "/tmp/" + filename
    data = result["output"]["data"]
    decode_data(data, save_path)
    print("Saved to " + save_path)
else:
    print("Generation failed :(")
    print(result)